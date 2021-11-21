import torch
import math
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class Time2Vec(nn.Module):
    def __init__(self, num_features: int, embedding_size: int):
        super().__init__()
        self.time2vec_dim = embedding_size - 1
        # trend
        self.wb = nn.Parameter(data=torch.empty(size=(num_features,)), requires_grad=True)
        self.bb = nn.Parameter(data=torch.empty(size=(num_features,)), requires_grad=True)

        # periodic
        self.wa = nn.Parameter(data=torch.empty(size=(1, num_features, self.time2vec_dim)), requires_grad=True)
        self.ba = nn.Parameter(data=torch.empty(size=(1, num_features, self.time2vec_dim)), requires_grad=True)

        self.wb.data.uniform_(-1, 1)
        self.bb.data.uniform_(-1, 1)
        self.wa.data.uniform_(-1, 1)
        self.ba.data.uniform_(-1, 1)

    def forward(self, inputs):
        bias = torch.mul(self.wb, inputs) + self.bb
        dp = torch.mul(torch.unsqueeze(inputs, -1), self.wa) + self.ba
        wgts = torch.sin(dp)

        ret = torch.cat([torch.unsqueeze(bias, -1), wgts], -1)
        ret = torch.reshape(ret, (-1, inputs.shape[1] * (self.time2vec_dim + 1)))
        return ret


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, sequence_length: int = 24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)[:,:d_model // 2]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), ].expand(x.shape)
        return self.dropout(x)


class TransformerBaseProps(LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.features_len = len(config.experiment.synop_train_features)
        self.embed_dim = self.features_len * (config.experiment.time2vec_embedding_size + 1)
        self.dropout = config.experiment.dropout
        self.use_pos_encoding = config.experiment.use_pos_encoding
        self.sequence_length = config.experiment.sequence_length
        self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.features_len, config.experiment.time2vec_embedding_size), batch_first=True)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)

        dense_layers = []
        features = self.embed_dim
        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)


class Transformer(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing

        self.n_heads = config.experiment.transformer_attention_heads
        self.ff_dim = config.experiment.transformer_ff_dim
        self.transformer_layers_num = config.experiment.transformer_attention_layers

        self.input_features_length = self.features_len

        if config.experiment.with_dates_inputs:
            self.input_features_length += 2
            self.embed_dim += 2 * (config.experiment.time2vec_embedding_size + 1)

        self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.input_features_length,
                                                                    config.experiment.time2vec_embedding_size),
                                                           batch_first=True)

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.transformer_layers_num, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, self.n_heads, self.ff_dim, self.dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.transformer_layers_num, decoder_norm)

        dense_layers = []
        features = self.embed_dim

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

    def forward(self, synop_inputs: torch.Tensor, synop_targets: torch.Tensor, epoch: int, stage=None,
                dates_embeddings: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) = None) -> torch.Tensor:
        if dates_embeddings is None:
            x = [synop_inputs]
        else:
            x = [synop_inputs, dates_embeddings[0], dates_embeddings[1]]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)

        if dates_embeddings is None:
            y = [synop_targets]
        else:
            y = [synop_targets, dates_embeddings[2], dates_embeddings[3]]

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)

        whole_target_embedding = torch.cat([*y, self.time_2_vec_time_distributed(torch.cat(y, -1))], -1)

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
                decoder_input = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), whole_target_embedding], 1)[:, first_taught:-1, ]
                decoder_input = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                target_mask = self.generate_mask(self.sequence_length - first_taught).to(self.device)
                next_pred = self.decoder(decoder_input, memory, tgt_mask=target_mask)
                output = next_pred if pred is None else torch.cat([pred, next_pred], 1)

            else:
                # non-gradual, just basic teacher forcing
                decoder_input = self.pos_encoder(whole_target_embedding) if self.use_pos_encoding else whole_target_embedding
                decoder_input = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), decoder_input], 1)[:, :-1, ]
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

        return torch.squeeze(self.classification_head_time_distributed(output), dim=-1)
