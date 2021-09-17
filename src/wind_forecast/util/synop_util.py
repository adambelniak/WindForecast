from datetime import timedelta

from tqdm import tqdm
import pandas as pd


def get_correct_dates_for_sequence(labels: pd.DataFrame, sequence_length: int, future_sequence_length: int,
                                   prediction_offset: int):
    """
    Takes DataFrame of synop observations and extracts these dates, which have consecutive sequence of length sequence_length + prediction_offset + future_sequence_length
    :param labels:
    :param sequence_length:
    :param future_sequence_length:
    :param target_param:
    :param prediction_offset:
    :return:
    """
    synop_dates = []
    one_hour = timedelta(hours=1)
    for date in tqdm(labels["date"]):
        next_local_date = date + one_hour

        if len(labels[labels["date"] == next_local_date]) > 0:
            # next frame exists, so the sequence is continued
            synop_dates.append(date)
        else:
            # there is no next frame, so the sequence is broken. Remove past frames
            for frame in range(1, sequence_length + future_sequence_length + prediction_offset):
                hours = timedelta(hours=frame)
                local_date_to_remove = date - hours
                synop_dates.remove(local_date_to_remove)

    return synop_dates
