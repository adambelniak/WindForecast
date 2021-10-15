import torch
import math
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerCMAX(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        n_heads = config.experiment.transformer_attention_heads
        ff_dim = config.experiment.transformer_ff_dim
        transformer_layers_num = config.experiment.transformer_attention_layers
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        conv_layers = []
        assert len(config.experiment.cnn_filters) > 0

        in_channels = 1
        for index, filters in enumerate(config.experiment.cnn_filters):
            out_channels = filters
            conv_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2),
                          padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels),
            ])
            if index != len(config.experiment.cnn_filters) - 1:
                conv_layers.append(nn.Dropout(config.experiment.dropout))
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers,
                                  nn.Flatten(),
                                  nn.Linear(in_features=conv_W * conv_H * out_channels, out_features=conv_W * conv_H * out_channels))
        self.conv_time_distributed = TimeDistributed(self.conv)

        self.embed_dim = self.features_len * (config.experiment.time2vec_embedding_size + 1) + conv_W * conv_H * out_channels

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers,
                                             encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(self.embed_dim, n_heads, ff_dim, self.dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, transformer_layers_num, decoder_norm)

        self.linear = nn.Linear(in_features=self.embed_dim, out_features=1)
        self.linear_time_distributed = TimeDistributed(self.linear, batch_first=True)
        self.flatten = nn.Flatten()

    def generate_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, inputs: torch.Tensor, cmax_inputs: torch.Tensor, targets: torch.Tensor, cmax_targets: torch.Tensor, epoch: int, stage=None) -> torch.Tensor:
        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        input_time_embedding = self.time_2_vec_time_distributed(inputs)
        whole_input_embedding = torch.cat([inputs, input_time_embedding, cmax_embeddings], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)
        self.conv_time_distributed.requires_grad_(False)
        cmax_targets_embeddings = self.conv_time_distributed(cmax_targets.unsqueeze(2))
        self.conv_time_distributed.requires_grad_(True)

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
                targets_time_embedding = self.time_2_vec_time_distributed(targets)
                targets = torch.cat([targets, targets_time_embedding, cmax_targets_embeddings], -1)
                y = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), targets], 1)[:, first_taught:-1, ]
                y = self.pos_encoder(y) if self.use_pos_encoding else y
                target_mask = self.generate_mask(self.sequence_length - first_taught).to(self.device)
                next_pred = self.decoder(y, memory, tgt_mask=target_mask)
                output = next_pred if pred is None else torch.cat([pred, next_pred], 1)

            else:
                # non-gradual, just basic teacher forcing
                targets_time_embedding = self.time_2_vec_time_distributed(targets)
                targets = torch.cat([targets, targets_time_embedding, cmax_targets_embeddings], -1)
                y = self.pos_encoder(targets) if self.use_pos_encoding else targets
                y = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), y], 1)[:, :-1, ]
                target_mask = self.generate_mask(self.sequence_length).to(self.device)
                output = self.decoder(y, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = torch.zeros(x.size(0), 1, self.embed_dim, device=self.device)  # SOS
            pred = None
            for frame in range(inputs.size(1)):
                y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                next_pred = self.decoder(y, memory)
                decoder_input = next_pred
                pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)
            output = pred

        return torch.squeeze(TimeDistributed(self.linear, batch_first=True)(output), dim=-1)
