import math

import torch as t

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.nbeatsx.nbeatsx import Nbeatsx
from wind_forecast.models.nbeatsx.nbeatsx_model import NBeatsx
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.common_util import get_pretrained_artifact_path, get_pretrained_state_dict


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

        self.n_insample_t += cmax_embed_dim

        block_list = self.create_stacks()

        self.model = NBeatsx(t.nn.ModuleList(block_list), classes=self.classes)

        if config.experiment.use_pretrained_artifact and type(self).__name__ is "Nbeatsx_CMAX":
            pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
            self.load_state_dict(get_pretrained_state_dict(pretrained_autoencoder_path))
            return

    def get_embeddings(self, batch):
        with_dates = self.config.experiment.with_dates_inputs
        with_gfs_params = self.use_gfs
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        dates_tensors = None if with_dates is False else batch[BatchKeys.DATES_TENSORS.value]

        cmax_input_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if with_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            input_elements = t.cat([synop_inputs, gfs_inputs], -1)
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

        input_elements = t.cat([input_elements, cmax_input_embeddings], -1)

        return input_elements, target_elements
