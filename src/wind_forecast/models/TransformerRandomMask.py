from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import Transformer


class TransformerRandomMask(Transformer):
    def __init__(self, config: Config):
        super().__init__(config)

    def generate_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length).uniform_() > 0.2) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask[:, 0] = 0.0  # first decoder input cannot be -inf
        return mask

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        all_synop_targets = batch[BatchKeys.ALL_SYNOP_TARGETS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_EMBEDDING.value]

        if self.config.experiment.with_dates_inputs is None:
            x = [synop_inputs, dates_embedding[0], dates_embedding[1]]
            y = [all_synop_targets, dates_embedding[2], dates_embedding[3]]

        else:
            x = [synop_inputs]
            y = [all_synop_targets]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)
        whole_target_embedding = torch.cat([*y, self.time_2_vec_time_distributed(torch.cat(y, -1))], -1)
        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)

        if stage in [None, 'fit']:
            y = self.pos_encoder(whole_target_embedding) if self.use_pos_encoding else whole_target_embedding
            y = torch.cat([self.getSOS(x.size(0)), y], 1)[:, :-1, ]
            target_mask = self.generate_mask(self.sequence_length).to(self.device)
            output = self.decoder(y, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = self.getSOS(x.size(0))
            pred = None
            for frame in range(synop_inputs.size(1)):
                y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                next_pred = self.decoder(y, memory)
                decoder_input = next_pred
                pred = next_pred if pred is None else torch.cat([pred, next_pred], 1)
            output = pred

        return torch.squeeze(self.classification_head_time_distributed(output), -1)
