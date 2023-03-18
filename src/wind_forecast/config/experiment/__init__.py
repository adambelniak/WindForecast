from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf.omegaconf import MISSING

# Experiment settings validation schema & default values

from synop.consts import SYNOP_TRAIN_FEATURES, SYNOP_PERIODIC_FEATURES, TEMPERATURE
from wind_forecast.util.common_util import NormalizationType


@dataclass
class ExperimentSettings:
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------

    # wandb tags
    _tags_: Optional[List[str]] = None

    # Seed for all random number generators
    seed: int = 1

    # Path to resume from. Two formats are supported:
    # - local checkpoints: path to checkpoint relative from run (results) directory
    # - wandb artifacts: wandb://ARTIFACT_PATH/ARTIFACT_NAME:VERSION@CHECKPOINT_NAME
    resume_checkpoint: Optional[str] = None

    # Path to pretrained artifact. Two formats are supported:
    # - local checkpoints: path to artifact relative from run (results) directory
    # - wandb artifacts: wandb://ENTITY/PROJECT_NAME/ARTIFACT_NAME:VERSION@CHECKPOINT_NAME
    pretrained_artifact: Optional[str] = 'wandb://mbelniak/wind-forecast-openstack/model-10kb6o2o:v0@model.ckpt'

    """
    Id of a run which should be the source of datamodule metadata use for training a pretrained model, e.g. mean and std
    """
    prediction_meta_run: Optional[str] = '10kb6o2o'

    # Same as above but for convolutional encoder for CMAX images
    pretrained_cmax_encoder: Optional[str] = 'wandb://mbelniak/wind-forecast-openstack-tune/model-1e6npo9p:v4@model.ckpt'

    # Enable checkpoint saving
    save_checkpoints: bool = True

    # Enable initial validation before training
    validate_before_training: bool = False

    check_val_every_n_epoch: int = 1

    # Run validation and test only
    skip_training: bool = False

    # Do not run validation
    skip_validation: bool = False

    # Do not run test
    skip_test: bool = False

    epochs: int = 40

    dropout: float = 0.5

    view_test_result: bool = True

    # ----------------------------------------------------------------------------------------------
    # Data loading settings
    # ----------------------------------------------------------------------------------------------

    # Training batch size
    batch_size: int = 128

    # Enable dataset shuffling
    shuffle: bool = True

    # Number of dataloader workers
    num_workers: int = 0

    # Model to use
    model: Any = MISSING

    system: Any = MISSING

    # ----------------------------------------------------------------------------------------------
    # Dataset specific settings
    # ----------------------------------------------------------------------------------------------

    # Lightning DataModule to use
    datamodule: Any = MISSING

    normalization_type: NormalizationType = NormalizationType.STANDARD

    cmax_normalization_type: NormalizationType = NormalizationType.MINMAX

    val_split: float = .2

    test_split: float = .2

    # classic or random
    dataset_split_mode: str = 'classic'

    # synop_file: str = "WARSZAWA-OKECIE_375_data.csv"
    synop_file: str = "WARSZAWA-OKECIE_352200375_data.csv"

    synop_station_code: str = "12375"

    tele_station_code: str = "352200375"

    synop_from_year: int = 2016

    cmax_from_year: int = 2016

    synop_to_year: int = 2022

    cmax_to_year: int = 2022

    # Dataset will load gfs data - it can be later used for plotting etc.
    load_gfs_data: bool = True

    # Dataset will load cmax data - it can be later used for plotting etc.
    load_cmax_data: bool = False

    # Model will load cmax data for future sequences as well
    load_future_cmax: bool = False

    # Model should use gfs data
    use_gfs_data: bool = False

    # Model should use cmax data
    use_cmax_data: bool = False

    dates_tensor_size: int = 2 # date in year and hour

    with_dates_inputs: bool = True

    stl_decompose: bool = False

    # ----------------------------------------------------------------------------------------------
    # Experiment specific settings
    # ----------------------------------------------------------------------------------------------

    # pass alone target on input to see if model is capable of reproducing it
    self_output_test: bool = False

    target_parameter: str = TEMPERATURE[1]

    differential_forecast: bool = False

    prediction_offset: int = 0

    data_dim_x: int = 53

    data_dim_y: int = 33

    train_parameters_config_file: str = "CommonGFSConfig.json"

    synop_train_features: List = field(default_factory=lambda: SYNOP_TRAIN_FEATURES)

    # Synop features which will be split into sin and cos during normalization phase
    synop_periodic_features: List = field(default_factory=lambda: SYNOP_PERIODIC_FEATURES)

    sequence_length: int = 24

    future_sequence_length: int = 24

    target_coords: Any = (52.1831174, 20.9875259)

    # e.g. for clouds - each octant is another class 0-9
    categorical_experiment: bool = False

    classes: int = 0

    # ----------------------------------------------------------------------------------------------
    # Model settings
    # ----------------------------------------------------------------------------------------------

    regressor_head_dims: List = field(default_factory=lambda: [64, 128, 32])

    gfs_on_head: bool = True

    cmax_projection_dim: int = 0

    use_pretrained_artifact: bool = False

    # ----------------------------------------------------------------------------------------------
    # CMAX autoencoder settings
    # ----------------------------------------------------------------------------------------------

    cnn_filters: List = field(default_factory=lambda: [16, 32, 32, 32, 32, 16])

    cmax_sample_size: Any = (900, 900)

    cmax_scaling_factor: int = 4

    cmax_h: int = 225

    cmax_w: int = 225

    use_pretrained_cmax_autoencoder: bool = False

    STD_scaling_factor: int = 5

    # ----------------------------------------------------------------------------------------------
    # LSTM settings
    # ----------------------------------------------------------------------------------------------

    lstm_num_layers: int = 4

    lstm_hidden_state: int = 512

    # ----------------------------------------------------------------------------------------------
    # TCN settings
    # ----------------------------------------------------------------------------------------------

    tcn_channels: List = field(default_factory=lambda: [32, 64])

    tcn_kernel_size: int = 3

    # Not used in S2S
    tcn_input_features: int = 1600

    emd_decompose: bool = False

    emd_decompose_trials: int = 10

    # ----------------------------------------------------------------------------------------------
    # Transformer settings
    # ----------------------------------------------------------------------------------------------

    teacher_forcing_epoch_num: int = 40

    gradual_teacher_forcing: bool = True

    transformer_ff_dim: int = 1024

    # not used - creating fixed embeddings resulted in much worse performance
    transformer_d_model: int = 256

    transformer_encoder_layers: int = 6

    transformer_decoder_layers: int = 6

    transformer_attention_heads: int = 1

    use_pos_encoding: bool = False

    # ----------------------------------------------------------------------------------------------
    # Spacetimeformer settings
    # ----------------------------------------------------------------------------------------------

    spacetimeformer_intermediate_downsample_convs: int = 0

    # ----------------------------------------------------------------------------------------------
    # NBeats settings
    # ----------------------------------------------------------------------------------------------

    # List of stack types.
    # Subset from ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet']
    nbeats_stack_types: List = field(default_factory=lambda: ['generic', 'exogenous_tcn'])

    # If True, repeats first block.
    nbeats_shared_weights: bool = False

    # Activation function.
    # An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
    nbeats_activation: str = 'selu'

    nbeats_num_blocks: List = field(default_factory=lambda: [8, 8])

    nbeats_num_layers: List = field(default_factory=lambda: [4, 4])

    nbeats_num_hidden: int = 32

    nbeats_expansion_coefficient_lengths: List = field(default_factory=lambda: [32])

    # Exogenous channels for non-interpretable exogenous basis.
    nbeats_exogenous_n_channels: int = 32

    # ----------------------------------------------------------------------------------------------
    # Arima settings
    # ----------------------------------------------------------------------------------------------

    arima_p: int = 1
    arima_d: int = 1
    arima_q: int = 1
    sarima_P: int = 0
    sarima_D: int = 1
    sarima_Q: int = 1
    sarima_M: int = 24

    # ----------------------------------------------------------------------------------------------
    # Linear settings
    # ----------------------------------------------------------------------------------------------

    linear_max_iter: int = 1000
    linear_L2_alpha: float = 1

    # ----------------------------------------------------------------------------------------------
    # Embedding settings
    # ----------------------------------------------------------------------------------------------

    time2vec_embedding_factor: int = 5

    value2vec_embedding_factor: int = 5

    use_time2vec: bool = False

    use_value2vec: bool = False

    # ----------------------------------------------------------------------------------------------
    # Other settings
    # ----------------------------------------------------------------------------------------------

    subregion_nlat: float = 53

    subregion_slat: float = 51

    subregion_elon: float = 22

    subregion_wlon: float = 20

    # pytorch_forecasting has a bug which calculates scaling in MASE based on future+future sequence instead of past+future sequence
    rescale_mase: bool = True