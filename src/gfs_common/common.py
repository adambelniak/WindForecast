import matplotlib.pyplot as plt
import numpy as np

from gfs_archive_0_25.gfs_processor.Coords import Coords

GFS_SPACE = Coords(56, 48, 13, 26)

GFS_PARAMETERS = [
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_1000"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_975"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_950"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_900"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_850"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_800"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_700"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_600"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_500"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_400"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_300"
    # },
    # {
    #     "name": "V GRD",
    #     "level": "ISBL_200"
    # },
    {
        "name": "V GRD",
        "level": "HTGL_10"
    },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_1000"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_975"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_950"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_900"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_850"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_800"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_700"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_600"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_500"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_400"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_300"
    # },
    # {
    #     "name": "U GRD",
    #     "level": "ISBL_200"
    # },
    {
        "name": "U GRD",
        "level": "HTGL_10"
    },
    {
        "name": "GUST",
        "level": "SFC_0"
    },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_1000"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_975"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_950"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_900"
    # },
    {
        "name": "TMP",
        "level": "ISBL_850"
    },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_800"
    # },
    {
        "name": "TMP",
        "level": "ISBL_700"
    },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_600"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_500"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_400"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_300"
    # },
    # {
    #     "name": "TMP",
    #     "level": "ISBL_200"
    # },
    {
        "name": "TMP",
        "level": "HTGL_2"
    },
    {
        "name": "CAPE",
        "level": "SFC_0"
    },
    # {
    #     "name": "CAPE",
    #     "level": "SPDL_0-180"
    # },
    # {
    #     "name": "CAPE",
    #     "level": "SPDL_0-255"
    # },
    # {
    #     "name": "LFT X",
    #     "level": "SFC_0"
    # },
    {
        "name": "DPT",
        "level": "HTGL_2"
    },
    # {
    #     "name": "CIN",
    #     "level": "SFC_0"
    # },
    # {
    #     "name": "CIN",
    #     "level": "SPDL_0-180"
    # },
    # {
    #     "name": "CIN",
    #     "level": "SPDL_0-255"
    # },
    # {
    #     "name": "P WAT",
    #     "level": "EATM_0"
    # },
    # {
    #     "name": "POT",
    #     "level": "SIGL_0.995"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_1000"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_975"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_950"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_900"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_850"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_800"
    # },
    {
        "name": "R H",
        "level": "ISBL_700"
    },
    # {
    #     "name": "R H",
    #     "level": "ISBL_600"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_500"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_400"
    # },
    # {
    #     "name": "R H",
    #     "level": "ISBL_300"
    # },
    {
        "name": "R H",
        "level": "HTGL_2"
    },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_1000"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_975"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_950"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_900"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_850"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_800"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_700"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_600"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_500"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_400"
    # },
    # {
    #     "name": "V VEL",
    #     "level": "ISBL_300"
    # },
    {
        "name": "T CDC",
        "level": "LCY_0"
    },
    {
        "name": "T CDC",
        "level": "MCY_0"
    },
    {
        "name": "T CDC",
        "level": "HCY_0"
    },
    {
        "name": "PRATE",
        "level": "SFC_0"
    },
    {
        "name": "PRES",
        "level": "SFC_0"
    },
    # {
    #     "name": "HLCY",
    #     "level": "HTGL_0-3000"
    # },
    # {
    #     "name": "HGT",
    #     "level": "ISBL_850"
    # },
    # {
    #     "name": "HGT",
    #     "level": "ISBL_700"
    # },
    {
        "name": "HGT",
        "level": "ISBL_500"
    },
    # {
    #     "name": "HGT",
    #     "level": "ISBL_300"
    # }
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
