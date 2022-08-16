from typing import Dict

import torch
import math
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.models.time2vec.Time2Vec import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), ].expand(x.shape)
        return self.dropout(x)


class TransformerEncoderBaseProps(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dropout = config.experiment.dropout
        self.use_pos_encoding = config.experiment.use_pos_encoding
        self.past_sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        self.time2vec_embedding_factor = config.experiment.time2vec_embedding_factor
        self.value2vec_embedding_factor = config.experiment.value2vec_embedding_factor
        self.use_time2vec = config.experiment.use_time2vec and config.experiment.with_dates_inputs
        self.use_value2vec = config.experiment.use_value2vec and self.value2vec_embedding_factor > 0

        if not self.use_value2vec:
            self.value2vec_embedding_factor = 0

        self.self_output_test = config.experiment.self_output_test
        if self.self_output_test:
            assert self.past_sequence_length == self.future_sequence_length, \
                "past_sequence_length must be equal future_sequence_length for self_output_test"

        self.n_heads = config.experiment.transformer_attention_heads
        self.ff_dim = config.experiment.transformer_ff_dim
        self.transformer_encoder_layers_num = config.experiment.transformer_encoder_layers

        self.transformer_head_dims = config.experiment.regressor_head_dims

        if self.self_output_test:
            self.features_length = 1
        else:
            self.features_length = len(config.experiment.synop_train_features) + len(config.experiment.synop_periodic_features)

        if self.use_time2vec and self.time2vec_embedding_factor == 0:
            self.time2vec_embedding_factor = self.features_length

        self.dates_dim = self.config.experiment.dates_tensor_size * self.time2vec_embedding_factor if self.use_time2vec \
            else self.config.experiment.dates_tensor_size * 2

        if self.use_time2vec:
            self.time_embed = TimeDistributed(Time2Vec(self.config.experiment.dates_tensor_size, self.time2vec_embedding_factor),
                                              batch_first=True)
        if self.use_value2vec:
            self.value_embed = TimeDistributed(Value2Vec(self.features_length, self.value2vec_embedding_factor),
                                               batch_first=True)

        self.embed_dim = self.features_length * (self.value2vec_embedding_factor + 1) + self.dates_dim
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.create_encoder()
        self.head_input_dim = self.embed_dim
        self.create_head()
        self.flatten = nn.Flatten()

    def create_encoder(self):
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.transformer_encoder_layers_num, encoder_norm)

    def create_head(self):
        dense_layers = []
        features = self.head_input_dim
        for neurons in self.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))

        self.regressor_head = nn.Sequential(*dense_layers)

    def generate_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerEncoderGFSBaseProps(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

        self.gfs_on_head = config.experiment.gfs_on_head
        gfs_params = process_config(config.experiment.train_parameters_config_file).params
        gfs_params_len = len(gfs_params)
        param_names = [x['name'] for x in gfs_params]
        if "V GRD" in param_names and "U GRD" in param_names:
            gfs_params_len += 1  # V and U will be expanded into velocity, sin and cos

        if not self.self_output_test:
            self.features_length += gfs_params_len

        if self.use_time2vec and self.time2vec_embedding_factor == 0:
            self.time2vec_embedding_factor = self.features_length

        self.dates_dim = config.experiment.dates_tensor_size * self.time2vec_embedding_factor if self.use_time2vec \
            else 2 * config.experiment.dates_tensor_size

        if self.use_value2vec:
            self.value_embed = TimeDistributed(Value2Vec(self.features_length, self.value2vec_embedding_factor),
                                               batch_first=True)

        self.embed_dim = self.features_length * (self.value2vec_embedding_factor + 1) + self.dates_dim
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.create_encoder()

        self.head_input_dim = self.embed_dim
        if self.gfs_on_head:
            self.head_input_dim += 1
        self.create_head()
        self.flatten = nn.Flatten()


class TransformerBaseProps(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.transformer_decoder_layers_num = self.config.experiment.transformer_decoder_layers
        self.create_decoder()

    def create_decoder(self):
        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout,
                                                   batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)

        self.decoder = nn.TransformerDecoder(decoder_layer, self.transformer_decoder_layers_num, decoder_norm)

    def inference(self, inference_length: int, decoder_input: torch.Tensor, memory: torch.Tensor):
        pred = None
        for frame in range(inference_length):
            y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
            next_pred = self.decoder(y, memory)
            decoder_input = torch.cat([decoder_input, next_pred[:, -1:, :]], -2)
            pred = decoder_input[:, 1:, :]
        return pred

    def masked_teacher_forcing(self, decoder_input: torch.Tensor, memory: torch.Tensor, mask_matrix_dim: int):
        target_mask = self.generate_mask(mask_matrix_dim).to(self.device)
        return self.decoder(decoder_input, memory, tgt_mask=target_mask)

    def base_transformer_forward(self, epoch, stage, input_embedding, target_embedding, memory):
        if epoch < self.teacher_forcing_epoch_num and stage not in ['test', 'predict', 'validate']:
            if self.gradual_teacher_forcing:
                # first - Teacher Forcing - masked targets as decoder inputs
                first_interfere_index = math.floor(epoch / self.teacher_forcing_epoch_num * target_embedding.size(1))
                decoder_input = torch.cat([input_embedding[:, -1:, :], target_embedding], 1)[:, :first_interfere_index, ]
                output = self.masked_teacher_forcing(decoder_input, memory, first_interfere_index)

                #then - inference
                output = self.inference(target_embedding.size(1) - first_interfere_index, output, memory)
            else:
                # non-gradual, just basic teacher forcing
                decoder_input = torch.cat([input_embedding[:, -1:, :], target_embedding], 1)[:, :-1, ]
                output = self.masked_teacher_forcing(decoder_input, memory, target_embedding.size(1))

        else:
            # inference - pass only predictions to decoder
            decoder_input = input_embedding[:, -1:, :]  # SOS
            output = self.inference(self.future_sequence_length, decoder_input, memory)
        return output


class TransformerGFSBaseProps(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.transformer_decoder_layers_num = self.config.experiment.transformer_decoder_layers
        self.create_decoder()

    def create_decoder(self):
        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout,
                                                   batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)

        self.decoder = nn.TransformerDecoder(decoder_layer, self.transformer_decoder_layers_num, decoder_norm)

    def inference(self, inference_length: int, decoder_input: torch.Tensor, memory: torch.Tensor):
        pred = None
        for frame in range(inference_length):  # do normal prediction for the beginning frames
            y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
            next_pred = self.decoder(y, memory)
            decoder_input = torch.cat([decoder_input, next_pred[:, -1:, :]], -2)
            pred = decoder_input[:, 1:, :]
        return pred

    def masked_teacher_forcing(self, decoder_input: torch.Tensor, memory: torch.Tensor, mask_matrix_dim: int):
        decoder_input = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
        target_mask = self.generate_mask(mask_matrix_dim).to(self.device)
        return self.decoder(decoder_input, memory, tgt_mask=target_mask)

    def base_transformer_forward(self, epoch: int, stage: str, input_embedding: torch.Tensor,
                                 target_embedding: torch.Tensor, memory: torch.Tensor):
        if epoch < self.teacher_forcing_epoch_num and stage not in ['test', 'predict', 'validate']:
            # Teacher forcing - masked targets as decoder inputs
            if self.gradual_teacher_forcing:
                # first - Teacher Forcing - masked targets as decoder inputs
                first_interfere_index = math.floor(epoch / self.teacher_forcing_epoch_num * target_embedding.size(1))
                decoder_input = torch.cat([input_embedding[:, -1:, :], target_embedding], 1)[:,
                                :first_interfere_index, ]
                output = self.masked_teacher_forcing(decoder_input, memory, first_interfere_index)

                # then - inference
                output = self.inference(target_embedding.size(1) - first_interfere_index, output, memory)
            else:
                # non-gradual, just basic teacher forcing
                decoder_input = torch.cat([input_embedding[:, -1:, :], target_embedding], 1)[:, :-1, ]
                output = self.masked_teacher_forcing(decoder_input, memory, target_embedding.size(1))

        else:
            # inference - pass only predictions to decoder
            decoder_input = input_embedding[:, -1:, :]  # SOS
            output = self.inference(self.future_sequence_length, decoder_input, memory)
        return output


class Transformer(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         False, is_train)

        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        if is_train:
            target_embedding = self.pos_encoder(target_elements) if self.use_pos_encoding else target_elements

        memory = self.encoder(input_embedding)
        output = self.base_transformer_forward(epoch, stage, input_embedding,
                                               target_embedding if is_train else None, memory)

        return torch.squeeze(self.regressor_head(self.forecaster(output)), -1)
