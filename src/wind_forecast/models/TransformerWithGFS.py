import math

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import Transformer, Time2Vec, PositionalEncoding
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TransformerWithGFS(Transformer):
    def __init__(self, config: Config):
        super().__init__(config)
        # TODO - return all gfs target from datamodule
        if config.experiment.use_all_gfs_as_input:
            gfs_params_len = len(process_config(config.experiment.train_parameters_config_file))
            self.input_features_length += gfs_params_len

            self.embed_dim += gfs_params_len * (config.experiment.time2vec_embedding_size + 1)

        self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.input_features_length,
                                                                    config.experiment.time2vec_embedding_size),
                                                           batch_first=True)

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.transformer_layers_num, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout,
                                                   batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.transformer_layers_num, decoder_norm)

        # TODO - return all gfs target from datamodule
        if config.experiment.with_dates_inputs:
            features = self.embed_dim + 3
        else:
            features = self.embed_dim + 1
        dense_layers = []

        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def get_embedding(self, gfs: torch.Tensor,
                      dates_embeddings: (torch.Tensor, torch.Tensor),
                      synop: torch.Tensor):
        if gfs is None:
            if dates_embeddings is None:
                x = [synop]
            else:
                x = [synop, dates_embeddings[0], dates_embeddings[1]]
        else:
            if dates_embeddings is None:
                x = [synop, gfs]
            else:
                x = [synop, gfs, dates_embeddings[0], dates_embeddings[1]]

        return torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)

    def forward(self, synop_inputs: torch.Tensor, gfs_inputs: torch.Tensor, gfs_targets: torch.Tensor,
                synop_targets: torch.Tensor, epoch: int, stage=None,
                dates_embeddings: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) = None) -> torch.Tensor:

        whole_input_embedding = self.get_embedding(gfs_inputs, dates_embeddings[0:2], synop_inputs)
        # TODO use gfs targets
        whole_target_embedding = self.get_embedding(gfs_inputs, dates_embeddings[2:4], synop_targets)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)

        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing - masked targets as decoder inputs
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                decoder_input = torch.zeros(x.size(0), 1, self.embed_dim, device=self.device)  # SOS
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                    next_pred = self.decoder(y, memory)
                    decoder_input = next_pred
                    pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)

                # then, do teacher forcing
                decoder_input = torch.cat(
                    [torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), whole_target_embedding], 1)[:,
                                first_taught:-1, ]
                decoder_input = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                target_mask = self.generate_mask(self.sequence_length - first_taught).to(self.device)
                next_pred = self.decoder(decoder_input, memory, tgt_mask=target_mask)
                output = next_pred if pred is None else torch.cat([pred, next_pred], 1)

            else:
                # non-gradual, just basic teacher forcing
                decoder_input = self.pos_encoder(
                    whole_target_embedding) if self.use_pos_encoding else whole_target_embedding
                decoder_input = torch.cat(
                    [torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), decoder_input], 1)[:, :-1, ]
                target_mask = self.generate_mask(self.sequence_length).to(self.device)
                output = self.decoder(decoder_input, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = torch.zeros(x.size(0), 1, self.embed_dim, device=self.device)  # SOS
            pred = None
            for frame in range(synop_inputs.size(1)):
                y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                next_pred = self.decoder(y, memory)
                decoder_input = next_pred
                pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)
            output = pred

        if dates_embeddings is None:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets], -1)), -1)
        else:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets, dates_embeddings[2], dates_embeddings[3]], -1)), -1)
