import torch

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import Transformer
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerRandomMask(Transformer):
    def __init__(self, config: Config):
        super().__init__(config)

    def generate_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length).uniform_() > 0.2) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask[:, 0] = 0.0  # first decoder input cannot be -inf
        return mask

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int, stage=None) -> torch.Tensor:
        input_time_embedding = self.time_2_vec_time_distributed(inputs)
        whole_input_embedding = torch.cat([inputs, input_time_embedding], -1)
        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)

        if stage in [None, 'fit']:
            targets_time_embedding = self.time_2_vec_time_distributed(targets)
            targets = torch.cat([targets, targets_time_embedding], -1)
            y = self.pos_encoder(targets) if self.use_pos_encoding else targets
            y = torch.cat([torch.zeros(x.size(0), 1, self.embed_dim, device=self.device), y], 1)[:, :-1, ]
            target_mask = self.generate_mask(self.sequence_length).to(self.device)
            output = self.decoder(y, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = torch.zeros(x.size(0), 1, self.embed_dim, device=self.device)  # SOS
            pred = None
            for frame in range(inputs.size(1)):
                y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                next_pred = self.decoder(y, memory)
                decoder_input = next_pred
                pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)
            output = pred

        return torch.squeeze(TimeDistributed(self.linear, batch_first=True)(output), dim=-1)
