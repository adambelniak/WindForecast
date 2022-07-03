from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import Transformer


# Not investigated and developed - performance not promising
class TransformerRandomMask(Transformer):
    def __init__(self, config: Config):
        super().__init__(config)

    def generate_mask(self, sequence_length: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sequence_length, sequence_length).uniform_() > 0.2) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask[:, 0] = 0.0  # first decoder input cannot be -inf
        return mask

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        all_synop_targets = batch[BatchKeys.SYNOP_FUTURE_X.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        whole_input_embedding = torch.cat([synop_inputs, self.simple_2_vec_time_distributed(synop_inputs)], -1)
        whole_target_embedding = torch.cat([all_synop_targets, self.simple_2_vec_time_distributed(all_synop_targets)], -1)

        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)
            whole_target_embedding = torch.cat([whole_target_embedding, *dates_tensors[1]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)

        if stage in [None, 'fit']:
            y = self.pos_encoder(whole_target_embedding) if self.use_pos_encoding else whole_target_embedding
            y = torch.cat([whole_input_embedding[:, -1:, :], y], 1)[:, :-1, ]
            target_mask = self.generate_mask(y.size(1)).to(self.device)
            output = self.decoder(y, memory, tgt_mask=target_mask)

        else:
            # inference - pass only predictions to decoder
            decoder_input = whole_input_embedding[:, -1:, :]
            pred = None
            for frame in range(self.future_sequence_length):
                y = self.pos_encoder(decoder_input) if self.use_pos_encoding else decoder_input
                next_pred = self.decoder(y, memory)
                decoder_input = torch.cat([decoder_input, next_pred[:, -1:, :]], -2)
                pred = decoder_input[:, 1:, :]
            output = pred

        return torch.squeeze(self.classification_head_time_distributed(output), -1)
