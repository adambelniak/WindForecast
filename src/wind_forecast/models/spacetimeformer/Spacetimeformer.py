from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.util.common_util import get_pretrained_artifact_path, get_pretrained_state_dict
from wind_forecast.util.config import process_config
from .extra_layers import ConvBlock, Normalization, FoldForPred
from .Encoder import Encoder, EncoderLayer
from .Decoder import Decoder, DecoderLayer
from .attention import (
    FullAttention,
    ProbAttention,
    AttentionLayer,
    PerformerAttention,
)
from .embed import Embedding

"""
d_yc: int = 1,
d_yt: int = 1,
d_x: int = 4,
max_seq_len: int = None,
attn_factor: int = 5,
d_model: int = 200,
d_queries_keys: int = 30,
d_values: int = 30,
n_heads: int = 8,
e_layers: int = 2,
d_layers: int = 3,
d_ff: int = 800,
start_token_len: int = 0,
time_emb_dim: int = 6,
dropout_emb: float = 0.1,
dropout_attn_matrix: float = 0.0,
dropout_attn_out: float = 0.0,
dropout_ff: float = 0.2,
dropout_qkv: float = 0.0,
pos_emb_type: str = "abs",
performer_attn_kernel: str = "relu",
performer_redraw_interval: int = 1000,
attn_time_windows: int = 1,
use_shifted_time_windows: bool = True,
embed_method: str = "spatio-temporal",
activation: str = "gelu",
norm: str = "batch",
use_final_norm: bool = True,
initial_downsample_convs: int = 0,
intermediate_downsample_convs: int = 0,
device = torch.device("cuda:0"),
null_value: float = None,
pad_value: float = None,
out_dim: int = None,
use_val: bool = True,
use_time: bool = True,
use_space: bool = True,
use_given: bool = True
"""

class Spacetimeformer(LightningModule):
    def __init__(
        self,
        config: Config
    ):
        super().__init__()
        self.config = config
        self.use_gfs = config.experiment.use_gfs_data
        self.gfs_on_head = config.experiment.gfs_on_head
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        self.features_length = len(config.experiment.synop_train_features) + len(config.experiment.synop_periodic_features)
        if config.experiment.stl_decompose:
            self.features_length *= 3
            self.features_length += 1  # + 1 for non-decomposed target param
        self.future_sequence_length = config.experiment.future_sequence_length
        assert self.future_sequence_length <= config.experiment.sequence_length

        if config.experiment.use_gfs_data:
            gfs_params = process_config(config.experiment.train_parameters_config_file).params
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded into velocity, sin and cos
            if config.experiment.stl_decompose:
                gfs_params_len = 3 * gfs_params_len + 1
            self.features_length += gfs_params_len

        self.transformer_encoder_layers_num = config.experiment.transformer_encoder_layers
        self.transformer_decoder_layers_num = config.experiment.transformer_decoder_layers

        assert config.experiment.spacetimeformer_intermediate_downsample_convs <= self.transformer_encoder_layers_num - 1
        split_length_into = self.features_length

        self.start_token_len = 0
        self.time_dim = (config.experiment.dates_tensor_size * config.experiment.time2vec_embedding_factor)\
            if config.experiment.use_time2vec else (2 * config.experiment.dates_tensor_size)

        self.token_dim = self.time_dim + ((config.experiment.value2vec_embedding_factor + 1) if config.experiment.use_value2vec else 1)

        # embeddings. seperate enc/dec in case the variable indices are not aligned
        self.enc_embedding = Embedding(
            n_x=self.features_length,
            n_time=config.experiment.dates_tensor_size if config.experiment.use_time2vec else self.time_dim,
            d_model=self.token_dim,
            time_emb_dim=config.experiment.time2vec_embedding_factor,
            value_emb_dim=config.experiment.value2vec_embedding_factor,
            start_token_len=self.start_token_len,
            is_encoder=True,
            use_val_embed=config.experiment.use_value2vec,
            use_time_embed=config.experiment.use_time2vec,
            use_position_emb=config.experiment.use_pos_encoding
        )
        self.dec_embedding = Embedding(
            n_x=self.features_length,
            n_time=config.experiment.dates_tensor_size if config.experiment.use_time2vec else self.time_dim,
            d_model=self.token_dim,
            time_emb_dim=config.experiment.time2vec_embedding_factor,
            value_emb_dim=config.experiment.value2vec_embedding_factor,
            start_token_len=self.start_token_len,
            is_encoder=False,
            use_val_embed=config.experiment.use_value2vec,
            use_time_embed=config.experiment.use_time2vec,
            use_position_emb=config.experiment.use_pos_encoding
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
                        config.experiment.spacetimeformer_attention_type,
                        **attn_kwargs,
                    ),
                    local_attention=self._attn_switch(
                        config.experiment.spacetimeformer_attention_type,
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
                        config.experiment.spacetimeformer_attention_type,
                        **attn_kwargs,
                    ),
                    local_self_attention=self._attn_switch(
                        config.experiment.spacetimeformer_attention_type,
                        **attn_kwargs,
                    ),
                    global_cross_attention=self._attn_switch(
                        config.experiment.spacetimeformer_attention_type,
                        **attn_kwargs,
                    ),
                    local_cross_attention=self._attn_switch(
                        config.experiment.spacetimeformer_attention_type,
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

        # final linear layers turn Transformer output into predictions
        # transform tokens into 1-dimensional outputs to allow folding them
        self.forecaster = nn.Linear(self.token_dim, 1, bias=True)
        features = self.features_length
        if self.use_gfs and self.gfs_on_head:
            features += 1

        dense_layers = []
        for neurons in config.experiment.regressor_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))

        self.regressor_head = nn.Sequential(*dense_layers)

        if config.experiment.use_pretrained_artifact and type(self).__name__ is "Spacetimeformer":
            pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
            self.load_state_dict(get_pretrained_state_dict(pretrained_autoencoder_path))
            return

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        # We have x as variables and y as target, but spacetimeformer has it the other way round. TODO redo it
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
            epoch=epoch,
            stage=stage,
            self_mask_seq=dec_mask_seq,
            cross_mask_seq=enc_dec_mask_seq,
            output_cross_attn=False
        )

        # forecasting predictions
        forecast_out = self.forecaster(dec_out)

        # fold flattened spatiotemporal format back into (batch, length, nx)
        forecast_out = FoldForPred(forecast_out, n_x=self.features_length)
        forecast_out = forecast_out[:, self.start_token_len : self.future_sequence_length, :]

        if self.use_gfs and self.gfs_on_head:
            gfs_preds = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return torch.squeeze(self.regressor_head(torch.cat([forecast_out, gfs_preds], -1)), -1)

        return torch.squeeze(self.regressor_head(forecast_out), -1)

    def get_embeddings(self, batch, is_train: bool):
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        if self.use_gfs:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            gfs_future = batch[BatchKeys.GFS_FUTURE_X.value].float()
            inputs = torch.cat([synop_inputs, gfs_inputs], -1)
            targets = torch.zeros((synop_inputs.shape[0], self.future_sequence_length, synop_inputs.shape[2])).to(inputs.device)
            targets = torch.cat([targets, gfs_future], -1)
        else:
            inputs = synop_inputs
            targets = torch.zeros((inputs.shape[0], self.future_sequence_length, inputs.shape[2])).to(inputs.device)

        # zero values for decoder input synop, but real values for gfs forecast
        dates = batch[BatchKeys.DATES_TENSORS.value]

        # embed context sequence
        enc_val_time_emb, _, enc_mask_seq = self.enc_embedding(input=inputs, dates=dates[0], cmax=None)
        # embed target context
        dec_val_time_emb, _, dec_mask_seq = self.dec_embedding(input=targets, dates=dates[1], cmax=None)

        return enc_val_time_emb, enc_mask_seq, dec_val_time_emb, dec_mask_seq

    def _attn_switch(
        self,
        attn_str: str,
        d_model: int,
        n_heads: int,
        d_qk: int,
        d_v: int,
        dropout_qkv: float,
        dropout_attn_matrix: float,
        attn_factor: int,
        performer_attn_kernel: str,
        performer_redraw_interval: int,
    ):

        if attn_str == "full":
            # standard full (n^2) attention
            Attn = AttentionLayer(
                attention=partial(FullAttention, attention_dropout=dropout_attn_matrix),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "prob":
            # Informer-style ProbSparse cross attention
            Attn = AttentionLayer(
                attention=partial(
                    ProbAttention,
                    factor=attn_factor,
                    attention_dropout=dropout_attn_matrix,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "performer":
            # Performer Linear Attention
            Attn = AttentionLayer(
                attention=partial(
                    PerformerAttention,
                    dim_heads=d_qk,
                    kernel=performer_attn_kernel,
                    feature_redraw_interval=performer_redraw_interval,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "none":
            Attn = None
        else:
            raise ValueError(f"Unrecognized attention str code '{attn_str}'")
        return Attn