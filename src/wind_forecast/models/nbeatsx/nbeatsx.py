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
    ExogenousBasisTCN, GenericBasis
from wind_forecast.models.nbeatsx.nbeatsx_model import NBeatsx, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis
from wind_forecast.models.time2vec.Time2Vec import Time2Vec
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
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
    GENERIC_BLOCK = 'generic'

    def __init__(self, config: Config):
        super(Nbeatsx, self).__init__()
        """
        N-BEATSx model.
        """

        self.config = config
        self.activation = config.experiment.nbeats_activation
        self.initialization = 'glorot_normal'
        if self.activation == 'selu': self.initialization = 'lecun_normal'

        # ------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.future_sequence_length = config.experiment.future_sequence_length
        self.past_sequence_length = config.experiment.sequence_length
        self.shared_weights = config.experiment.nbeats_shared_weights
        self.stack_types = config.experiment.nbeats_stack_types
        self.n_blocks = config.experiment.nbeats_num_blocks
        self.n_layers = config.experiment.nbeats_num_layers
        self.n_hidden = [layers * [config.experiment.nbeats_num_hidden] for layers in self.n_layers]
        self.n_harmonics = 0
        self.n_polynomials = 0
        self.exogenous_n_channels = config.experiment.nbeats_exogenous_n_channels
        self.tcn_channels = config.experiment.tcn_channels

        # Regularization and optimization parameters
        self.batch_normalization = True
        self.dropout = config.experiment.dropout
        self.use_gfs = config.experiment.use_gfs_data
        self.time2vec_embedding_factor = config.experiment.time2vec_embedding_factor
        self.value2vec_embedding_factor = config.experiment.value2vec_embedding_factor
        self.use_time2vec = config.experiment.use_time2vec and config.experiment.with_dates_inputs
        self.use_value2vec = config.experiment.use_value2vec

        # No static features in our case
        self.x_static_n_hidden, self.x_static_n_inputs = 0, 0

        if self.use_gfs:
            gfs_params = process_config(config.experiment.train_parameters_config_file).params
            n_gfs_features = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                n_gfs_features += 1  # V and U will be expanded int velocity, sin and cos

            self.n_insample_t = len(config.experiment.synop_train_features) + n_gfs_features + len(
                config.experiment.synop_periodic_features)
            self.n_outsample_t = n_gfs_features
        else:
            self.n_insample_t = len(config.experiment.synop_train_features) + len(
                config.experiment.synop_periodic_features)
            self.n_outsample_t = 0

        if self.use_time2vec and self.time2vec_embedding_factor == 0:
            self.time2vec_embedding_factor = self.features_length

        self.dates_dim = self.config.experiment.dates_tensor_size * self.time2vec_embedding_factor if self.use_time2vec \
            else self.config.experiment.dates_tensor_size * 2

        if self.use_time2vec:
            self.time_embed = TimeDistributed(Time2Vec(self.config.experiment.dates_tensor_size,
                                                       self.time2vec_embedding_factor), batch_first=True)
        if self.use_value2vec:
            self.value2vec_insample = TimeDistributed(Value2Vec(self.n_insample_t,
                                                                self.value2vec_embedding_factor), batch_first=True)
            self.value2vec_outsample = TimeDistributed(Value2Vec(self.n_outsample_t,
                                                                 self.value2vec_embedding_factor), batch_first=True)
            self.n_insample_t += self.n_insample_t * self.value2vec_embedding_factor
            self.n_outsample_t += self.n_outsample_t * self.value2vec_embedding_factor

        if config.experiment.with_dates_inputs:
            self.n_insample_t = self.n_insample_t + self.dates_dim
            self.n_outsample_t = self.n_outsample_t + self.dates_dim if self.use_gfs else 0

        block_list = self.create_stacks()

        self.model = NBeatsx(t.nn.ModuleList(block_list))

    def create_stacks(self):
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
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
                                                   theta_n_dim=4 * int(
                                                       np.ceil(self.n_harmonics / 2 * self.future_sequence_length) - (
                                                               self.n_harmonics - 1)),
                                                   basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                          backcast_size=self.past_sequence_length,
                                                                          forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'trend':
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
                                                   theta_n_dim=2 * (self.n_polynomials + 1),
                                                   basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                    backcast_size=self.past_sequence_length,
                                                                    forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'identity':
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
                                                   theta_n_dim=self.past_sequence_length + self.future_sequence_length,
                                                   basis=IdentityBasis(backcast_size=self.past_sequence_length,
                                                                       forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'generic':
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
                                                   theta_n_dim=self.past_sequence_length + self.future_sequence_length,
                                                   basis=GenericBasis(backcast_size=self.past_sequence_length,
                                                                      forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous':
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
                                                   theta_n_dim=self.n_insample_t + self.n_outsample_t,
                                                   basis=ExogenousBasisInterpretable(),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_tcn':
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
                                                   theta_n_dim=2 * self.tcn_channels[-1],
                                                   basis=ExogenousBasisTCN(self.n_insample_t,
                                                                           self.n_outsample_t,
                                                                           self.tcn_channels,
                                                                           dropout_prob=self.dropout,
                                                                           theta_n_dim=2 * self.tcn_channels[-1],
                                                                           forecast_size=self.future_sequence_length),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=list(self.n_hidden[i]),
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_wavenet':
                        nbeats_block = NBeatsBlock(T=self.past_sequence_length,
                                                   x_static_n_inputs=self.x_static_n_inputs,
                                                   x_static_n_hidden=self.x_static_n_hidden,
                                                   n_insample_t=self.n_insample_t,
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
        insample_elements, outsample_elements = self.get_embeddings(batch)
        synop_past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float()

        # No static features in my case
        return self.model(x_static=t.Tensor([]), insample_y=synop_past_targets,
                          insample_x_t=insample_elements.permute(0, 2, 1),
                          outsample_x_t=outsample_elements.permute(0, 2, 1) if self.use_gfs else None)

    def get_embeddings(self, batch):
        with_dates = self.config.experiment.with_dates_inputs
        with_gfs_params = self.use_gfs
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()

        dates_tensors = None if with_dates is False else batch[BatchKeys.DATES_TENSORS.value]

        if with_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            input_elements = t.cat([synop_inputs, gfs_inputs], -1)
            all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()
            target_elements = all_gfs_targets
        else:
            input_elements = synop_inputs

        value_embed = [self.value2vec_insample, self.value2vec_outsample] if self.use_value2vec else None
        time_embed = self.time_embed if self.use_time2vec else None

        if value_embed is not None:
            input_elements = t.cat([input_elements, value_embed[0](input_elements)], -1)
            if with_gfs_params:
                target_elements = t.cat([target_elements, value_embed[1](target_elements)], -1)

        if with_dates:
            if time_embed is not None:
                input_elements = t.cat([input_elements, time_embed(dates_tensors[0])], -1)
            else:
                input_elements = t.cat([input_elements, dates_tensors[0]], -1)

            if with_gfs_params:
                if time_embed is not None:
                    target_elements = t.cat([target_elements, time_embed(dates_tensors[1])], -1)
                else:
                    target_elements = t.cat([target_elements, dates_tensors[1]], -1)

        return input_elements, target_elements if with_gfs_params else None
