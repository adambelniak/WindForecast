import matplotlib.pyplot as plt
import numpy as np


GFS_PARAMETERS = [
    ("V GRD", "ISBL_975"),
    ("V GRD", "ISBL_950"),
    ("V GRD", "ISBL_750"),
    ("V GRD", "ISBL_500"),
    ("V GRD", "ISBL_450"),
    ("V GRD", "ISBL_400"),
    ("V GRD", "ISBL_300"),
    ("V GRD", "ISBL_250"),
    ("V GRD", "ISBL_200"),
    ("V GRD", "HTGL_10"),
    ("U GRD", "ISBL_925")
]


def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def convert_wind(single_date_series, u_wind_label, v_wind_label):
    velocity = np.sqrt(single_date_series[u_wind_label] ** 2 + single_date_series[v_wind_label] ** 2)

    return velocity