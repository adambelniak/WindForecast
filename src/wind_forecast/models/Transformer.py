import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config


class Time2Vec(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.time2vec_dim = config.experiment.time2vec_embedding_size - 1
        # trend
        self.wb = nn.Parameter(data=torch.empty(size=(len(config.experiment.synop_train_features),)), requires_grad=True)
        self.bb = nn.Parameter(data=torch.empty(size=(len(config.experiment.synop_train_features),)), requires_grad=True)

        # periodic
        self.wa = nn.Parameter(data=torch.empty(size=(1, len(config.experiment.synop_train_features), self.time2vec_dim)), requires_grad=True)
        self.ba = nn.Parameter(data=torch.empty(size=(1, len(config.experiment.synop_train_features), self.time2vec_dim)), requires_grad=True)

        self.wb.data.uniform_(-1, 1)
        self.bb.data.uniform_(-1, 1)
        self.wa.data.uniform_(-1, 1)
        self.ba.data.uniform_(-1, 1)

    def forward(self, inputs):
        bias = torch.mul(self.wb, inputs) + self.bb
        dp = torch.mul(inputs.reshape(*inputs.shape, 1), self.wa) + self.ba
        wgts = torch.sin(dp)

        ret = torch.cat([torch.unsqueeze(bias, -1), wgts], -1)
        ret = torch.reshape(ret, (-1, inputs.shape[1] * (self.time2vec_dim + 1)))
        return ret


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class AttentionBlock(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        num_heads = config.experiment.transformer_attention_heads
        # kdim = config.experiment.transformer_attention_kdim
        # vdim = config.experiment.transformer_attention_vdim
        ff_dim = config.experiment.transformer_ff_dim
        features_len = len(config.experiment.synop_train_features)
        embed_dim = features_len * (config.experiment.time2vec_embedding_size + 1)
        dropout = config.experiment.dropout

        self.attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, kdim=embed_dim, vdim=embed_dim, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

        self.ff_conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=ff_dim, kernel_size=1)
        self.act = nn.ReLU()
        self.ff_conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=embed_dim, kernel_size=1)

        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, inputs):
        x, _ = self.attention(inputs, inputs, inputs)
        x = self.attention_dropout(x)
        attention_out = self.attention_norm(inputs + x)

        x = attention_out
        x = x.reshape((x.shape[0], x.shape[-1], x.shape[-2]))
        x = self.ff_conv1(x)
        x = self.act(x)
        x = self.ff_conv2(x)
        x = self.act(x)
        x = self.ff_dropout(x)
        x = x.reshape((x.shape[0], x.shape[-1], x.shape[-2]))
        x = self.ff_norm(attention_out + x)

        return x


class Transformer(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        embed_dim = len(config.experiment.synop_train_features) * (config.experiment.time2vec_embedding_size + 1)
        self.time2vec = Time2Vec(config)
        self.attention_layers = [
            AttentionBlock(config) for _ in range(config.experiment.transformer_attention_layers)]
        self.linear = nn.Linear(in_features=embed_dim * config.experiment.sequence_length, out_features=1)

    def forward(self, inputs):
        time_embedding = TimeDistributed(self.time2vec, batch_first=True)(inputs)
        x = torch.cat([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            if self.device.type is not attention_layer.device.type:
                # device in AttentionBlock is not being set and I coulnd't make it work :(
                attention_layer.to(self.device)
            x = attention_layer(x)

        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2]))  # flat vector of features out

        return torch.squeeze(self.linear(x))

