import math
from typing import Dict

import torch as t

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.nbeatsx.nbeatsx import Nbeatsx
from wind_forecast.models.nbeatsx.nbeatsx_model import NBeatsx
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class Nbeatsx_CMAX(Nbeatsx):

    def __init__(self, config: Config):
        super().__init__(config)
        assert config.experiment.use_cmax_data, "use_cmax_data should be True for nbeatx_cmax experiment"
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_channels = config.experiment.cnn_filters[-1]
        self.cmax_conv = CMAXEncoder(config)
        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.cmax_conv, config)
        self.conv_time_distributed = TimeDistributed(self.cmax_conv, batch_first=True)

        cmax_embed_dim = conv_W * conv_H * out_channels

        if self.use_gfs:
            embeddable_input_features = self.synop_features_length + self.gfs_features_length + cmax_embed_dim
            embeddable_output_features = self.gfs_features_length
        else:
            embeddable_input_features = self.synop_features_length + cmax_embed_dim
            embeddable_output_features = 0

        if self.use_value2vec:
            self.value2vec_insample = TimeDistributed(Value2Vec(embeddable_input_features,
                                                                self.value2vec_embedding_factor), batch_first=True)
            self.value2vec_outsample = TimeDistributed(Value2Vec(embeddable_output_features,
                                                                 self.value2vec_embedding_factor), batch_first=True)

        self.n_insample_t += cmax_embed_dim

        block_list = self.create_stacks()

        self.model = NBeatsx(t.nn.ModuleList(block_list))

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
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        dates_tensors = None if with_dates is False else batch[BatchKeys.DATES_TENSORS.value]

        cmax_input_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if with_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            input_elements = t.cat([synop_inputs, gfs_inputs, cmax_input_embeddings], -1)
            all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()
            target_elements = all_gfs_targets
        else:
            input_elements = synop_inputs
            target_elements = None

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

        return input_elements, target_elements
