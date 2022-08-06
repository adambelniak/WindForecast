import math
from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.spacetimeformer.Spacetimeformer import Spacetimeformer
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from .Decoder import Decoder, DecoderLayer
from .Encoder import Encoder, EncoderLayer
from .embed import Embedding
from .extra_layers import ConvBlock, Normalization, FoldForPred


class Spacetimeformer_cmax(Spacetimeformer):
    def __init__(
        self,
        config: Config
    ):
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
        self.features_length += cmax_embed_dim

        split_length_into = self.features_length

        # embeddings. seperate enc/dec in case the variable indices are not aligned
        self.enc_embedding = Embedding(
            d_input=self.features_length,
            d_time_features=config.experiment.dates_tensor_size if config.experiment.use_time2vec else self.time_dim,
            d_model=self.token_dim,
            time_emb_dim=config.experiment.time2vec_embedding_factor,
            value_emb_dim=config.experiment.value2vec_embedding_factor,
            downsample_convs=0,
            method='spatio-temporal',
            null_value=None,
            start_token_len=self.start_token_len,
            is_encoder=True,
            position_emb='t2v',
            max_seq_len=None,
            data_dropout=None,
            use_val_embed=config.experiment.use_value2vec,
            use_time_embed=config.experiment.use_time2vec,
            use_space=True,
            use_given=True,
            use_position_emb=config.experiment.use_pos_encoding,
            emb_dropout=0.0
        )
        self.dec_embedding = Embedding(
            d_input=self.features_length,
            d_time_features=config.experiment.dates_tensor_size if config.experiment.use_time2vec else self.time_dim,
            d_model=self.token_dim,
            time_emb_dim=config.experiment.time2vec_embedding_factor,
            value_emb_dim=config.experiment.value2vec_embedding_factor,
            downsample_convs=0,
            method='spatio-temporal',
            null_value=None,
            start_token_len=self.start_token_len,
            is_encoder=False,
            position_emb='t2v',
            max_seq_len=None,
            data_dropout=None,
            use_val_embed=config.experiment.use_value2vec,
            use_time_embed=config.experiment.use_time2vec,
            use_space=True,
            use_given=True,
            use_position_emb=config.experiment.use_pos_encoding,
            emb_dropout=0.0
        )

        # Select Attention Mechanisms
        attn_kwargs = {
            "d_model": self.token_dim,
            "n_heads": config.experiment.transformer_attention_heads,
            "d_qk": 20,
            "d_v": 20,
            "dropout_qkv": 0.0,
            "dropout_attn_matrix": 0.0,
            "attn_factor": 5,
            "performer_attn_kernel": 'relu',
            "performer_redraw_interval": 1000,
        }

        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    global_attention=self._attn_switch(
                        'performer',
                        **attn_kwargs,
                    ),
                    local_attention=self._attn_switch(
                        'performer',
                        **attn_kwargs,
                    ),
                    d_model=self.token_dim,
                    d_yc=self.features_length,
                    time_windows=1,
                    # encoder layers alternate using shifted windows, if applicable
                    time_window_offset=2 if (l % 2 == 1) else 0,
                    d_ff=config.experiment.transformer_ff_dim,
                    dropout_ff=config.experiment.dropout,
                    dropout_attn_out=0.0,
                    activation='relu',
                    norm='batch',
                )
                for l in range(self.transformer_encoder_layers_num)
            ],
            conv_layers=[
                ConvBlock(split_length_into=split_length_into, d_model=self.token_dim)
                for l in range(config.experiment.spacetimeformer_intermediate_downsample_convs)
            ],
            norm_layer=Normalization('batch', d_model=self.token_dim)
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    global_self_attention=self._attn_switch(
                        'performer',
                        **attn_kwargs,
                    ),
                    local_self_attention=self._attn_switch(
                        'performer',
                        **attn_kwargs,
                    ),
                    global_cross_attention=self._attn_switch(
                        'performer',
                        **attn_kwargs,
                    ),
                    local_cross_attention=self._attn_switch(
                        'performer',
                        **attn_kwargs,
                    ),
                    d_model=self.token_dim,
                    time_windows=1,
                    # decoder layers alternate using shifted windows, if applicable
                    time_window_offset=2 if (l % 2 == 1) else 0,
                    d_ff=config.experiment.transformer_ff_dim,
                    # temporal embedding effectively has 1 variable
                    # for the purposes of time windowing.
                    d_yt=self.features_length,
                    d_yc=self.features_length,
                    dropout_ff=config.experiment.dropout,
                    dropout_attn_out=config.experiment.dropout,
                    activation='relu',
                    norm='batch',
                )
                for l in range(self.transformer_decoder_layers_num)
            ],
            norm_layer=Normalization('batch', d_model=self.token_dim)
        )

        features = self.features_length
        if self.use_gfs:
            features += 1

        dense_layers = []
        for neurons in config.experiment.classification_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))

        self.classification_head = nn.Sequential(*dense_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        enc_vt_emb, enc_mask_seq, dec_vt_emb, dec_mask_seq = self.get_embeddings(batch, is_train)

        # encode context sequence
        enc_out, enc_self_attns = self.encoder(
            val_time_space_emb=enc_vt_emb,
            self_mask_seq=enc_mask_seq,
            output_attn=False
        )

        if enc_mask_seq is not None:
            enc_dec_mask_seq = enc_mask_seq.clone()
        else:
            enc_dec_mask_seq = enc_mask_seq

        # decode target sequence w/ encoded context
        dec_out, dec_cross_attns = self.decoder(
            val_time_space_emb=dec_vt_emb,
            cross=enc_out,
            self_mask_seq=dec_mask_seq,
            cross_mask_seq=enc_dec_mask_seq,
            output_cross_attn=False
        )

        # forecasting predictions
        forecast_out = self.forecaster(dec_out)

        # fold flattened spatiotemporal format back into (batch, length, d_yt)
        forecast_out = FoldForPred(forecast_out, dy=self.features_length)
        forecast_out = forecast_out[:, self.start_token_len : self.future_sequence_length, :]

        if self.use_gfs:
            gfs_preds = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return torch.squeeze(self.classification_head(torch.cat([forecast_out, gfs_preds], -1)), -1)

        return torch.squeeze(self.classification_head(forecast_out), -1)

    def get_embeddings(self, batch, is_train: bool):
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        cmax_input_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if self.use_gfs:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()

        dates = batch[BatchKeys.DATES_TENSORS.value]

        if self.use_gfs:
            inputs = torch.cat([synop_inputs, gfs_inputs, cmax_input_embeddings], -1)
        else:
            inputs = torch.cat([synop_inputs, cmax_input_embeddings], -1)
        # embed context sequence
        enc_val_time_emb, _, enc_mask_seq = self.enc_embedding(input=inputs, dates=dates[0])

        # embed target context
        targets = torch.zeros_like(inputs)
        dec_val_time_emb, _, dec_mask_seq = self.dec_embedding(input=targets, dates=dates[1])

        return enc_val_time_emb, enc_mask_seq, dec_val_time_emb, dec_mask_seq