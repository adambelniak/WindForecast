import datetime
import os
import pandas as pd
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

register_matplotlib_converters()
sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

def decompose_synop(synop_file):
    if not os.path.exists(synop_file):
        raise Exception(f"CSV file with synop data does not exist at path {synop_file}.")

    data = prepare_synop_dataset(synop_file, ['temperature'], norm=False,
                                 dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=2016, to_year=2017)

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

    series = data[["date", "temperature"]]
    series = series[series['date'] <= datetime.datetime(year=2016, month=4, day=1)]
    series = series.set_index('date')
    stl = STL(series, seasonal=11, period=24, trend=43, low_pass=25)
    res = stl.fit(inner_iter=1, outer_iter=10)
    O, T, S, R = res.observed, res.trend, res.seasonal, res.resid
    fig, axes = plt.subplots(4, 1, figsize=(15, 5), sharex=True)
    fig.suptitle('Dekompozycja szeregu metodą STL')
    axes[0].set(ylabel='y')
    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    plt.setp(axes[2].get_xticklabels(), visible=False)

    axes[1].set(ylabel='T')
    axes[2].set(ylabel='S')
    axes[3].set(ylabel='R', xlabel='Data')

    sns.lineplot(ax=axes[0], data=O, x='date', y='temperature')
    sns.lineplot(ax=axes[1], data=T)
    sns.lineplot(ax=axes[2], data=S)
    sns.lineplot(ax=axes[3], data=R)
    plt.show()


if __name__ == "__main__":
    decompose_synop(os.path.join(SYNOP_DATASETS_DIRECTORY, 'WARSZAWA-OKECIE_352200375_data.csv'))