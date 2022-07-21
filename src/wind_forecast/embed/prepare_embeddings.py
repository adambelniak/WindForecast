from wind_forecast.consts import BatchKeys
import torch


def get_embeddings(batch, with_dates, time_embed, value_embed, with_gfs_params, with_future):
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()

        if with_future:
            all_synop_targets = batch[BatchKeys.SYNOP_FUTURE_X.value].float()
        dates_tensors = None if with_dates is False else batch[BatchKeys.DATES_TENSORS.value]

        if with_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            input_elements = torch.cat([synop_inputs, gfs_inputs], -1)
            if with_future:
                all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()
                target_elements = torch.cat([all_synop_targets, all_gfs_targets], -1)
        else:
            input_elements = synop_inputs
            if with_future:
                target_elements = all_synop_targets

        if value_embed is not None:
            input_elements = torch.cat([input_elements, value_embed(input_elements)], -1)
            if with_future:
                target_elements = torch.cat([target_elements, value_embed(target_elements)], -1)

        if with_dates:
            if time_embed is not None:
                input_elements = torch.cat([input_elements, time_embed(dates_tensors[0])], -1)
            else:
                input_elements = torch.cat([input_elements, dates_tensors[0]], -1)

            if with_future:
                if time_embed is not None:
                    target_elements = torch.cat([target_elements, time_embed(dates_tensors[1])], -1)
                else:
                    target_elements = torch.cat([target_elements, dates_tensors[1]], -1)

        return input_elements, target_elements if with_future else None