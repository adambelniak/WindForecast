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

from functools import partial
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch as t

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.nbeatsx.nbeatsx_model import ExogenousBasisInterpretable, ExogenousBasisWavenet, \
    ExogenousBasisTCN
from wind_forecast.models.nbeatsx.nbeatsx_model import NBeatsx, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis
from wind_forecast.util.config import process_config


def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass
        else:
            assert 1 < 0, f'Initialization {initialization} not found'


class Nbeatsx(pl.LightningModule):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self, config: Config):
        super(Nbeatsx, self).__init__()
        """
        N-BEATSx model.
        """

        self.activation = config.experiment.nbeats_activation
        self.initialization = 'glorot_normal'
        if self.activation == 'selu': self.initialization = 'lecun_normal'

        # ------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.future_sequence_length = config.experiment.future_sequence_length
        self.input_size = config.experiment.sequence_length
        self.shared_weights = config.experiment.nbeats_shared_weights
        self.stack_types = config.experiment.nbeats_stack_types
        self.n_blocks = config.experiment.nbeats_num_blocks
        self.n_layers = config.experiment.nbeats_num_layers
        self.n_hidden = config.experiment.nbeats_num_hidden
        self.n_harmonics = 0
        self.n_polynomials = 0
        self.exogenous_n_channels = config.experiment.nbeats_exogenous_n_channels

        # Regularization and optimization parameters
        self.batch_normalization = True
        self.dropout = config.experiment.dropout
        # No static features in our case
        self.x_s_n_hidden, self.n_x_s = 0, 0

        n_gfs_features = len(process_config(config.experiment.train_parameters_config_file))
        self.n_insample_t = len(config.experiment.synop_train_features) + 1 + n_gfs_features
        self.n_outsample_t = n_gfs_features

        block_list = self.create_stack()

        self.model = NBeatsx(t.nn.ModuleList(block_list))

    def create_stack(self):
        x_t_n_inputs = self.input_size

        # ------------------------ Model Definition ------------------------#
        block_list = []
        for i in range(len(self.stack_types)):
            for block_id in range(self.n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list) == 0) and self.batch_normalization:
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if self.shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if self.stack_types[i] == 'seasonality':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=4 * int(
                                                       np.ceil(self.n_harmonics / 2 * self.future_sequence_length) - (
                                                                   self.n_harmonics - 1)),
                                                   basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                          backcast_size=self.input_size,
                                                                          forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'trend':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.n_polynomials + 1),
                                                   basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                    backcast_size=self.input_size,
                                                                    forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'identity':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=self.input_size + self.future_sequence_length,
                                                   basis=IdentityBasis(backcast_size=self.input_size,
                                                                       forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.n_x_t,
                                                   basis=ExogenousBasisInterpretable(),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_tcn':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.exogenous_n_channels,
                                                   basis=ExogenousBasisTCN(self.exogenous_n_channels, self.n_insample_t, self.n_outsample_t, dropout_prob=self.dropout),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_wavenet':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.exogenous_n_channels,
                                                   basis=ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    else:
                        assert 1 < 0, f'Block type not found!'
                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=self.initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, batch: Dict[str, t.Tensor], epoch: int, stage=None) -> t.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float().permute(0, 2, 1)
        synop_past_targets = batch[BatchKeys.SYNOP_PAST_TARGETS.value].float()
        gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float().permute(0, 2, 1)
        gfs_all_targets = batch[BatchKeys.ALL_GFS_TARGETS.value].float().permute(0, 2, 1)

        # No static features in my case
        return self.model(x_s=t.Tensor([]), insample_y=synop_past_targets,
                          insample_x_t=t.cat([synop_inputs, gfs_inputs], 1), outsample_x_t=gfs_all_targets)
