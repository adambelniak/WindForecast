import torch

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerEncoderGFSBaseProps

# TODO check if working after refactoring
class Transformer(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, synop_input, gfs_input):
        time_embedding = self.simple_2_vec_time_distributed(synop_input)
        x = torch.cat([synop_input, time_embedding], -1)
        x = self.pos_encoder(x) if self.use_pos_encoding else x
        x = self.encoder(x)
        x = self.flatten(x)  # flat vector of synop_features out

        return torch.squeeze(self.classification_head(x))

