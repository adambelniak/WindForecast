from sklearn.utils import shuffle

from models.dense_model import create_model
import argparse
import numpy as np
from preprocess.synop_preprocess import prepare_synop_dataset
from preprocess.gfs_preprocess import prepare_gfs_data
import tqdm
from datetime import datetime, timedelta
from sklearn import preprocessing

GFS_CSV_DIR = 'gfs-archive-0-25/gfs_processor/output_csvs/54_6-18_8'


def convert_wind(single_date_series, u_wind_label, v_wind_label):
    velocity = np.sqrt(single_date_series[u_wind_label] ** 2 + single_date_series[v_wind_label] ** 2)

    return velocity


def add_hours_based_on_time_shift(date):
    if date.month > 10 or date.month < 4:
        return date + timedelta(hours=1)
    else:
        return date + timedelta(hours=2)


def prepare_single_hour_data(single_gfs, synop_dataset, parameter, specific_hour):
    condition = single_gfs.date.str.contains(specific_hour)
    single_gfs_spec_hour = single_gfs[condition]
    if len(single_gfs_spec_hour.index) == 1:
        # don't take first two rows, because clouds are not correct
        if single_gfs.index[condition] in [0, 1]:
            return None, None
        else:
            single_gfs_spec_hour = single_gfs_spec_hour.iloc[0]
    elif len(single_gfs_spec_hour.index) == 0:
        return None, None
    else:
        single_gfs_spec_hour = single_gfs_spec_hour.iloc[1]
    if len(single_gfs_spec_hour.index) != 9:
        return None, None
    single_gfs_spec_hour['velocity'] = convert_wind(single_gfs_spec_hour, 'U GRD_HTGL:10', 'V GRD_HTGL:10')
    date = datetime.strptime(single_gfs_spec_hour['date'], '%Y-%m-%dT%H:%M:%S')
    date = add_hours_based_on_time_shift(date)
    single_gfs_spec_hour.drop(['date', 'U GRD_HTGL:10', 'V GRD_HTGL:10'], inplace=True)
    single_gfs_spec_hour['date'] = abs(date.month - 6.5)

    vals_in_gfs = single_gfs_spec_hour.to_numpy()
    for val in vals_in_gfs:
        if np.math.isnan(val):
            return None, None

    return vals_in_gfs, np.array([synop_dataset[synop_dataset['date'] == date].drop('date', axis=1).iloc[0][parameter]])


def prepare_gfs_data_for_forecast_frame(single_gfs, synop_dataset, parameter, forecast_frame: int):
    if len(single_gfs.index) < forecast_frame:
        return None, None
    single_gfs = single_gfs[2:forecast_frame]
    gfs_result = []
    labels_result = []
    for i in range(2, forecast_frame):
        velocity = convert_wind(single_gfs.loc[i], 'U GRD_HTGL:10', 'V GRD_HTGL:10')
        date = datetime.strptime(single_gfs.loc[i]['date'], '%Y-%m-%dT%H:%M:%S')
        date = add_hours_based_on_time_shift(date)
        vals_in_gfs = single_gfs.drop(['date', 'U GRD_HTGL:10', 'V GRD_HTGL:10'], axis=1).loc[i]
        vals_in_gfs['velocity'] = velocity
        if len(vals_in_gfs.index) != 7:
            continue
        vals_in_gfs = vals_in_gfs.to_numpy()
        for val in vals_in_gfs:
            if np.math.isnan(val):
                continue
        gfs_result.append(vals_in_gfs)
        labels_result.append(np.array([synop_dataset.loc[synop_dataset['date'] == date].drop('date', axis=1).iloc[0][parameter]]))

    return gfs_result, labels_result


def prepare_data(gfs_data_dir: str, synop_data_path: str, parameter: str, specific_hour: str, forecast_frame: int):
    print("Creating data set")
    synop_dataset = prepare_synop_dataset(synop_data_path, False)

    if parameter not in synop_dataset.columns:
        raise Exception("Parameter {} is not available in synop dataset.".format(parameter))

    gfs_dataset_by_forecast_date = prepare_gfs_data(gfs_data_dir)

    gfs_dataset_input, dataset_labels = [], []
    for key in tqdm.tqdm(gfs_dataset_by_forecast_date.keys()):
        single_gfs = gfs_dataset_by_forecast_date[key]

        if specific_hour != '':  # train only for the specific hour of the first day in forecast
            gfs_data, labels_data = prepare_single_hour_data(single_gfs, synop_dataset, parameter, specific_hour)
            if gfs_data is None:
                continue
            gfs_dataset_input.append(gfs_data)
            dataset_labels.append(labels_data)
        else:
            gfs_data, labels_data = prepare_gfs_data_for_forecast_frame(single_gfs, synop_dataset, parameter, forecast_frame)
            if gfs_data is None:
                continue
            gfs_dataset_input.extend(gfs_data)
            dataset_labels.extend(labels_data)

    print(gfs_dataset_input[:5])
    print(dataset_labels[:5])
    return gfs_dataset_input, dataset_labels


def train_model(**kwargs):
    dataset_input, dataset_labels = prepare_data(kwargs['gfs_data_dir'], kwargs['synop_data_csv'], kwargs['parameter'],
                                                 kwargs['spec_hour'], kwargs['forecast_frame'])

    label_min_val, label_max_val = min(dataset_labels), max(dataset_labels)
    print("Labels max value: {}, min value: {}".format(label_max_val, label_min_val))
    label_scaler = 1 / (label_max_val - label_min_val)

    gfs_temperatures = [row[3] for row in dataset_input]
    gfs_min_val, gfs_max_val = min(gfs_temperatures), max(gfs_temperatures)
    print("GFS feature max value: {}, min value: {}".format(gfs_max_val, gfs_min_val))
    gfs_scaler = 1 / (gfs_max_val - gfs_min_val)

    sum = 0
    for index in range(0, len(dataset_input) - 1):
        sum = sum + abs(dataset_input[index][3] - 273 - dataset_labels[index][0])

    print("GFS average error: {}".format(sum / len(dataset_input)))

    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_input, dataset_labels = min_max_scaler.fit_transform(dataset_input), min_max_scaler.fit_transform(dataset_labels)
    dataset_input, dataset_labels = shuffle(dataset_input, dataset_labels)

    model = create_model()
    model.fit(x=dataset_input, y=dataset_labels, epochs=20, batch_size=64, validation_split=kwargs['train_split'])
    # plot_history(history)
    Xnew = np.array([[0,0,0,(274 - gfs_min_val) * gfs_scaler,0,1,0,1]])
    # make a prediction
    ynew = model.predict(Xnew)
    # show the inputs and predicted outputs
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
    print(f"Predicted scaled: {ynew[0] / label_scaler + label_min_val}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gfs_data_dir', help='Directory with GFS forecasts in csv format', default=GFS_CSV_DIR)
    parser.add_argument('--synop_data_csv', help='CSV with SYNOP data', default='preprocess/synop_data/135_data.csv')
    parser.add_argument('--train_split', help='Train split factor from 0 to 1', default=0.1, type=float)
    parser.add_argument('--parameter', help='Parameter to train on', default='temperature', type=str)
    parser.add_argument('--spec_hour', help='Train only for a specific hour of a first day in forecast', default='', type=str)
    parser.add_argument('--forecast_frame', help='How many steps of GFS forecast should be taken into account. '
                                                 'Applied only if spec_hour is not specified', default=8, type=int)
    args = parser.parse_args()
    train_model(**vars(args))
