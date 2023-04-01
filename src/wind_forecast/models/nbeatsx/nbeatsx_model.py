"""
MIT License

Copyright (c) 2021 Kin G. Olivares, Cristian Challu, Grzegorz Marcjasz, Rafał Weron and Artur Dubrawski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
from typing import Tuple, Union, List

import numpy as np
import torch as t
import torch.nn as nn

from wind_forecast.models.nbeatsx.tcn import TemporalConvNet


class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self, T: int, x_static_n_inputs: int, x_static_n_hidden: int, n_insample_x: int, theta_n_dim: int,
                 basis: nn.Module, n_layers: int, theta_n_hidden: List[int], batch_normalization: bool,
                 dropout_prob: float, activation: str, classes: int = 0):
        """
        """
        super().__init__()

        if x_static_n_inputs == 0:
            x_static_n_hidden = 0

        self.classes = classes

        if self.classes > 0:
            theta_n_hidden = [T * self.classes + x_static_n_hidden + T * n_insample_x] + theta_n_hidden
        else:
            theta_n_hidden = [T + x_static_n_hidden + T * n_insample_x] + theta_n_hidden

        self.x_static_n_inputs = x_static_n_inputs
        self.x_static_n_hidden = x_static_n_hidden
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):

            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i + 1]))
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i + 1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim * classes if classes > 0 else theta_n_dim)]
        layers = hidden_layers + output_layer

        # x_static_n_inputs is computed with data, x_static_n_hidden is provided by user, if 0 no statics are used
        if (self.x_static_n_inputs > 0) and (self.x_static_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=x_static_n_inputs, out_features=x_static_n_hidden)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_static: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # Static exogenous
        if (self.x_static_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_static = self.static_encoder(x_static)
            insample_y = t.cat((insample_y, x_static), 1)

        # Compute local projection weights and projection
        theta = self.layers(t.cat([insample_y,
                                   insample_x_t.contiguous().view(insample_x_t.size()[0],
                                                                  insample_x_t.size()[1] * insample_x_t.size()[2])], -1))
        if self.classes > 0:
            sizes = theta.size()
            theta = theta.reshape(sizes[0], sizes[1] // self.classes, self.classes)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast


class NBeatsx(nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: nn.ModuleList, classes=0):
        super().__init__()
        self.blocks = blocks
        self.classes = classes

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_static: t.Tensor, return_decomposition=False) -> Union[
        t.Tensor, Tuple[t.Tensor, t.Tensor]]:

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        block_forecasts = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_static=x_static)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_time)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        if self.classes > 0:
            block_forecasts = block_forecasts.reshape(*block_forecasts.size()[0:2], block_forecasts.size()[2] // self.classes, self.classes)
            forecast = forecast.reshape(forecast.size()[0], forecast.size()[1] // self.classes, self.classes)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, classes=0):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.classes = classes
        if classes > 0:
            self.forecast_linear = nn.Linear(in_features=(backcast_size + forecast_size) * classes, out_features=forecast_size * classes)
            self.backcast_linear = nn.Linear(in_features=(backcast_size + forecast_size) * classes, out_features=backcast_size * classes)
        else:
            self.forecast_linear = nn.Linear(in_features=backcast_size + forecast_size, out_features=forecast_size)
            self.backcast_linear = nn.Linear(in_features=backcast_size + forecast_size, out_features=backcast_size)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        if self.classes > 0:
            backcast = self.backcast_linear(theta.reshape(theta.size()[0], theta.size()[1] * theta.size()[2]))
            forecast = self.forecast_linear(theta.reshape(theta.size()[0], theta.size()[1] * theta.size()[2]))
        else:
            backcast = self.backcast_linear(theta)
            forecast = self.forecast_linear(theta)
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(np.zeros(1, dtype=np.float32),
                              np.arange(harmonics, harmonics / 2 * forecast_size,
                                        dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

        backcast_cos_template = t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32)
        backcast_sin_template = t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32)
        backcast_template = t.cat([backcast_cos_template, backcast_sin_template], dim=0)

        forecast_cos_template = t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32)
        forecast_sin_template = t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32)
        forecast_template = t.cat([forecast_cos_template, forecast_sin_template], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ExogenousBasisWavenet(nn.Module):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(t.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        padding = (kernel_size - 1) * (2 ** 0)
        input_layer = [nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=kernel_size, padding=padding, dilation=2 ** 0),
                       Chomp1d(padding),
                       nn.ReLU(),
                       nn.Dropout(dropout_prob)]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(nn.Conv1d(in_channels=out_features, out_channels=out_features,
                                         padding=padding, kernel_size=3, dilation=dilation))
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.Sequential(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = x_t * self.weight  # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class ExogenousBasisTCN(nn.Module):
    def __init__(self, insample_features, outsample_features, tcn_channels: List[int], kernel_size=3,
                 dropout_prob=0, theta_n_dim=0, forecast_size=0, classes=0):
        super().__init__()

        self.classes = classes
        self.insample_tcn = TemporalConvNet(num_inputs=insample_features, num_channels=tcn_channels,
                                            kernel_size=kernel_size,
                                            dropout=dropout_prob)
        if outsample_features > 0:
            self.outsample_tcn = TemporalConvNet(num_inputs=outsample_features, num_channels=tcn_channels,
                                                 kernel_size=kernel_size,
                                                 dropout=dropout_prob)
        else:
            # no outsample features, so use generic basis for forecast
            assert theta_n_dim > 0, "theta_n_dim must be bigger than 0 if no outsample features available"
            assert forecast_size > 0, "forecast_size must be bigger than 0 if no outsample features available"
            self.outsample_tcn = nn.Linear(in_features=theta_n_dim // 2 * classes, out_features=forecast_size * classes)

    def transform(self, insample_x_t, outsample_x_t):
        backcast_basis = self.insample_tcn(insample_x_t)[:]
        forecast_basis = self.outsample_tcn(outsample_x_t)[:]
        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        if outsample_x_t is not None:
            backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
            cut_point = forecast_basis.shape[1]
            if self.classes > 0:
                backcast = t.einsum('bpc,bpt->btc', theta[:, cut_point:], backcast_basis)
                forecast = t.einsum('bpc,bpt->btc', theta[:, :cut_point], forecast_basis)
                backcast = backcast.reshape(backcast.size()[0], backcast.size()[1] * backcast.size()[2])
                forecast = forecast.reshape(forecast.size()[0], forecast.size()[1] * forecast.size()[2])
            else:
                backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
                forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        else:
            cut_point = theta.size(1) // 2
            if self.classes > 0:
                backcast_basis, forecast_basis = self.transform(insample_x_t, theta.reshape(theta.size()[0], theta.size()[1] * theta.size()[2])[:, :cut_point])
                backcast = t.einsum('bpc,bpt->btc', theta[:, cut_point:], backcast_basis)
            else:
                backcast_basis, forecast_basis = self.transform(insample_x_t, theta[:, :cut_point])
                backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
            forecast = forecast_basis

        return backcast, forecast


class ExogenousBasisLSTM(nn.Module):
    def __init__(self, insample_features: int, outsample_features: int, lstm_num_layers: int, lstm_hidden_state: int,
                    dropout=0, theta_n_dim=0, forecast_size=0):
        super().__init__()
        self.use_exogenous_forecast = outsample_features > 0
        self.encoder_lstm = nn.LSTM(input_size=insample_features, hidden_size=lstm_hidden_state, batch_first=True,
                                    dropout=dropout, num_layers=lstm_num_layers,
                                    proj_size=insample_features - outsample_features)

        if self.use_exogenous_forecast:
            self.outsample_net = nn.LSTM(input_size=insample_features, hidden_size=lstm_hidden_state,
                                    batch_first=True,
                                    dropout=dropout, num_layers=lstm_num_layers,
                                    proj_size=insample_features - outsample_features)
        else:
            # no outsample features, so use generic basis for forecast
            assert theta_n_dim > 0, "theta_n_dim must be bigger than 0 if no outsample features available"
            assert forecast_size > 0, "forecast_size must be bigger than 0 if no outsample features available"
            self.outsample_net = nn.Linear(in_features=insample_features, out_features=forecast_size)

    def transform(self, insample_x_t, outsample_x_t):
        insample_x_t = insample_x_t.permute(0, 2, 1)
        outsample_x_t = outsample_x_t.permute(0, 2, 1)
        backcast_basis, state = self.encoder_lstm(insample_x_t)
        if self.use_exogenous_forecast:
            forecast_basis = self.decoder_forward(state, insample_x_t, outsample_x_t)
        else:
            forecast_basis = self.outsample_net(outsample_x_t)
        return backcast_basis.permute(0, 2, 1), forecast_basis.permute(0, 2, 1)

    def decoder_forward(self, state, input_elements, decoder_elements):
        # inference - pass only predictions to decoder
        decoder_input = t.cat(
            [
                input_elements[:, -1:, :-(decoder_elements.size()[-1])],
                decoder_elements[:, :1, :]
            ], -1)
        pred = None
        future_seq_len = decoder_elements.size()[-2]
        for frame in range(future_seq_len):
            next_pred, state = self.outsample_net(decoder_input, state)
            pred = t.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
            if frame < future_seq_len - 1:
                decoder_input = t.cat(
                    [
                        next_pred[:, -1:, :],
                        decoder_elements[:, (frame + 1):(frame + 2), :]
                    ], -1)
        output = pred

        return output

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        if outsample_x_t is not None:
            backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
            cut_point = forecast_basis.shape[1]
            backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
            forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        else:
            cut_point = theta.size(1) // 2
            backcast_basis, forecast_basis = self.transform(insample_x_t, theta[:, :cut_point])
            backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
            forecast = forecast_basis

        return backcast, forecast
