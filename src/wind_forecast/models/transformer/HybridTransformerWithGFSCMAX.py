import math
from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.transformer.HybridTransformerWithGFS import HybridTransformerWithGFS
from wind_forecast.models.transformer.Transformer import PositionalEncoding
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class HybridTransformerWithGFSCMAX(HybridTransformerWithGFS):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_channels = config.experiment.cnn_filters[-1]
        self.conv = CMAXEncoder(config)
        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv, config)
        self.conv_time_distributed = TimeDistributed(self.conv, batch_first=True)

        self.embed_dim += conv_W * conv_H * out_channels
        self.decoder_output_dim += conv_W * conv_H * out_channels
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.create_encoder()
        self.create_decoder()
        self.decoder_projection = torch.nn.Linear(in_features=self.embed_dim, out_features=self.decoder_output_dim)
        self.head_input_dim = self.decoder_output_dim
        if self.gfs_on_head:
            self.head_input_dim += 1
        self.create_head()

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, all_synop_targets, all_gfs_targets, cmax_future, future_dates = \
            self.get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                self.time_embed if self.use_time2vec else None, is_train)

        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements

        memory = self.encoder(input_embedding)

        output = self.decoder_forward_cmax(epoch, stage, memory, input_embedding, all_synop_targets, all_gfs_targets,
                                      cmax_future, future_dates)

        if self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return torch.squeeze(self.regressor_head(torch.cat([output, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(output), -1)

    def decoder_forward_cmax(self, epoch: int, stage: str, encoder_output, input_embedding: torch.Tensor,
                        all_synop_targets, all_gfs_targets, cmax_future_embeddings, future_dates):
        first_decoder_input = torch.cat(
            [
                input_embedding[:, -1:, :-(self.gfs_embed_dim + self.dates_dim)],
                all_gfs_targets[:, :1, :],
                future_dates[:, :1, :]
            ], -1)
        if epoch < self.teacher_forcing_epoch_num and stage not in ['test', 'predict', 'validate']:
            # Teacher forcing - masked targets as decoder inputs
            if self.gradual_teacher_forcing:
                # first - Teacher Forcing - masked targets as decoder inputs
                first_infer_index = self.future_sequence_length - min(
                    math.floor(epoch * self.future_sequence_length / self.teacher_forcing_epoch_num),
                    self.future_sequence_length)

                if first_infer_index > 0:
                    next_decoder_inputs = torch.cat(
                        [
                            all_synop_targets,
                            cmax_future_embeddings,
                            all_gfs_targets,
                            future_dates
                        ], -1)
                    decoder_input = torch.cat([first_decoder_input, next_decoder_inputs], 1)[:, :first_infer_index, ]
                    output = self.masked_teacher_forcing(decoder_input, encoder_output, first_infer_index)
                else:
                    output = None
                if output is None:
                    inference_input = first_decoder_input
                else:
                    inference_input = torch.cat([first_decoder_input, torch.cat([output[:, :, :],
                                                                                 all_gfs_targets[:, :first_infer_index,
                                                                                 :],
                                                                                 future_dates[:, :first_infer_index,
                                                                                 :]], -1)], -2)
                # then - inference
                output = self.hybrid_inference(self.future_sequence_length - first_infer_index, inference_input,
                                               encoder_output,
                                               all_gfs_targets, future_dates)
            else:
                # non-gradual, just basic teacher forcing
                next_decoder_inputs = torch.cat(
                    [
                        all_synop_targets,
                        cmax_future_embeddings,
                        all_gfs_targets,
                        future_dates
                    ], -1)
                decoder_input = torch.cat([first_decoder_input, next_decoder_inputs], 1)[:, :-1, ]
                output = torch.cat([first_decoder_input[:, -1:, :],
                                    self.masked_teacher_forcing(decoder_input, encoder_output,
                                                                self.future_sequence_length)], 1)

        else:
            # inference - pass only predictions to decoder
            decoder_input = first_decoder_input
            output = self.hybrid_inference(self.future_sequence_length, decoder_input, encoder_output,
                                                all_gfs_targets, future_dates)
        return output

    def get_embeddings(self, batch, with_dates, time_embed, with_future):
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        all_synop_targets = batch[BatchKeys.SYNOP_FUTURE_X.value].float() if with_future else None
        all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()

        cmax_past = batch[BatchKeys.CMAX_PAST.value].float()
        cmax_future = batch[BatchKeys.CMAX_FUTURE.value].float() if with_future else None
        cmax_past_embeddings = self.conv_time_distributed(cmax_past.unsqueeze(2))
        self.conv_time_distributed.requires_grad_(False)
        cmax_future_embeddings = self.conv_time_distributed(
            cmax_future.unsqueeze(2)) if cmax_future is not None else None
        self.conv_time_distributed.requires_grad_(True)

        dates_tensors = None if with_dates is False else batch[BatchKeys.DATES_TENSORS.value]
        future_dates = None

        gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
        input_elements = torch.cat([synop_inputs, gfs_inputs], -1)

        if self.use_value2vec:
            input_elements = torch.cat([input_elements, self.value_embed(input_elements)], -1)
            if with_future:
                all_synop_targets = torch.cat([all_synop_targets, self.value_embed_synop(all_synop_targets)], -1)
            all_gfs_targets = torch.cat([all_gfs_targets, self.value_embed_gfs(all_gfs_targets)], -1)

        input_elements = torch.cat([input_elements, cmax_past_embeddings], -1)

        if with_dates:
            if time_embed is not None:
                input_elements = torch.cat([input_elements, time_embed(dates_tensors[0])], -1)
            else:
                input_elements = torch.cat([input_elements, dates_tensors[0]], -1)

            if time_embed is not None:
                future_dates = time_embed(dates_tensors[1])
            else:
                future_dates = dates_tensors[1]

        return input_elements, all_synop_targets, all_gfs_targets, cmax_future_embeddings, future_dates
