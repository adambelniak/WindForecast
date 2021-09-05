import torch
import math
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.TransformerEncoder import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


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


class Transformer(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        features_len = len(config.experiment.synop_train_features)
        self.embed_dim = features_len * (config.experiment.time2vec_embedding_size + 1)
        self.time2vec = Time2Vec(config)
        self.sequence_length = config.experiment.sequence_length
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        dropout = config.experiment.dropout
        n_heads = config.experiment.transformer_attention_heads
        ff_dim = config.experiment.transformer_ff_dim
        transformer_layers_num = config.experiment.transformer_attention_layers

        self.pos_encoder = PositionalEncoding(self.embed_dim, dropout, self.sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, n_heads, ff_dim, dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, transformer_layers_num, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, n_heads, ff_dim, dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, transformer_layers_num, decoder_norm)

        self.linear = nn.Linear(in_features=self.embed_dim, out_features=1)

    def generate_square_subsequent_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int, stage=None) -> torch.Tensor:
        input_time_embedding = TimeDistributed(self.time2vec, batch_first=True)(inputs)
        x = torch.cat([inputs, input_time_embedding], -1)
        # x = self.pos_encoder(inputs)
        memory = self.encoder(x)

        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing - masked targets as decoder inputs
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                decoder_input = torch.zeros(x.size(0), 1, self.embed_dim, device=self.device)  # SOS
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    y = self.pos_encoder(decoder_input)
                    next_pred = self.decoder(y, memory)
                    decoder_input = next_pred
                    pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)

                # then, do teacher forcing
                targets_time_embedding = TimeDistributed(self.time2vec, batch_first=True)(targets)
                targets = torch.cat([targets, targets_time_embedding], -1)
                y = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), targets], 1)[:, first_taught:-1, ]
                # y = self.pos_encoder(targets_shifted[:, first_taught:, :])
                target_mask = self.generate_square_subsequent_mask(self.sequence_length - first_taught).to(self.device)
                next_pred = self.decoder(y, memory, tgt_mask=target_mask)
                output = next_pred if pred is None else torch.cat([pred, next_pred], 1)

            else:
                # non-gradual, just basic teacher forcing
                # y = self.pos_encoder(targets)
                targets_time_embedding = TimeDistributed(self.time2vec, batch_first=True)(targets)
                targets = torch.cat([targets, targets_time_embedding], -1)
                y = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), targets], 1)[:, :-1, ]
                target_mask = self.generate_square_subsequent_mask(self.sequence_length).to(self.device)
                output = self.decoder(y, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = torch.zeros(x.size(0), 1, self.embed_dim, device=self.device)  # SOS
            pred = None
            for frame in range(inputs.size(1)):
                # y = self.pos_encoder(decoder_input)
                next_pred = self.decoder(decoder_input, memory)
                decoder_input = next_pred
                pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)
            output = pred

        return torch.squeeze(TimeDistributed(self.linear, batch_first=True)(output))
