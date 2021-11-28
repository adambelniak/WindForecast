import math
from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import PositionalEncoding
from wind_forecast.models.TransformerCMAX import TransformerCMAX
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TransformerCMAXWithGFS(TransformerCMAX):
    def __init__(self, config: Config):
        super().__init__(config)
        if config.experiment.use_all_gfs_params:
            gfs_params_len = len(process_config(config.experiment.train_parameters_config_file))
            self.input_features_length += gfs_params_len
            self.embed_dim += gfs_params_len * (config.experiment.time2vec_embedding_size + 1)

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers,
                                             encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.transformer_layers_num, decoder_norm)

        features = self.embed_dim
        dense_layers = []

        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def generate_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        all_synop_targets = batch[BatchKeys.ALL_SYNOP_TARGETS.value].float()
        gfs_targets = batch[BatchKeys.GFS_TARGETS.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_INPUTS.value].float()
        cmax_targets = batch[BatchKeys.CMAX_TARGETS.value].float()

        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_EMBEDDING.value]

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                all_gfs_targets = batch[BatchKeys.ALL_GFS_TARGETS.value].float()
                x = [synop_inputs, gfs_inputs, dates_embedding[0], dates_embedding[1]]
                y = [all_synop_targets, all_gfs_targets, dates_embedding[2], dates_embedding[3]]
            else:
                x = [synop_inputs, dates_embedding[0], dates_embedding[1]]
                y = [all_synop_targets, dates_embedding[2], dates_embedding[3]]
        else:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                all_gfs_targets = batch[BatchKeys.ALL_GFS_TARGETS.value].float()
                x = [synop_inputs, gfs_inputs]
                y = [all_synop_targets, all_gfs_targets]
            else:
                x = [synop_inputs]
                y = [all_synop_targets]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1)), cmax_embeddings], -1)

        self.conv_time_distributed.requires_grad_(False)
        cmax_targets_embeddings = self.conv_time_distributed(cmax_targets.unsqueeze(2))
        self.conv_time_distributed.requires_grad_(True)

        whole_target_embedding = torch.cat([*y, self.time_2_vec_time_distributed(torch.cat(y, -1)), cmax_targets_embeddings], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)

        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing - masked targets as decoder inputs
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                decoder_input = self.getSOS(x.size(0))
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                    next_pred = self.decoder(y, memory)
                    decoder_input = torch.cat([decoder_input, next_pred[:, -1:, :]], -2)
                    pred = decoder_input[:, 1:, :]

                # then, do teacher forcing
                # SOS is appended for case when first_taught is 0
                decoder_input = torch.cat([self.getSOS(x.size(0)), whole_target_embedding], 1)[:, first_taught:-1, ]
                decoder_input = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                target_mask = self.generate_mask(self.sequence_length - first_taught).to(self.device)
                next_pred = self.decoder(decoder_input, memory, tgt_mask=target_mask)
                output = next_pred if pred is None else torch.cat([pred, next_pred], 1)

            else:
                # non-gradual, just basic teacher forcing
                decoder_input = self.pos_encoder(
                    whole_target_embedding) if self.use_pos_encoding else whole_target_embedding
                decoder_input = torch.cat([self.getSOS(x.size(0)), decoder_input], 1)[:, :-1, ]
                target_mask = self.generate_mask(self.sequence_length).to(self.device)
                output = self.decoder(decoder_input, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = self.getSOS(x.size(0))
            pred = None
            for frame in range(synop_inputs.size(1)):
                y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                next_pred = self.decoder(y, memory)
                decoder_input = torch.cat([decoder_input, next_pred[:, -1:, :]], -2)
                pred = decoder_input[:, 1:, :]
            output = pred

        return torch.squeeze(self.classification_head_time_distributed(output), -1)