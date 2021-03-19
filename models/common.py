import matplotlib.pyplot as plt
import numpy as np


GFS_PARAMETERS = [
    ("V GRD", "ISBL_1000"),
    ("V GRD", "ISBL_975"),
    ("V GRD", "ISBL_950"),
    ("V GRD", "ISBL_900"),
    ("V GRD", "ISBL_850"),
    ("V GRD", "ISBL_800"),
    ("V GRD", "ISBL_700"),
    ("V GRD", "ISBL_600"),
    ("V GRD", "ISBL_500"),
    ("V GRD", "ISBL_400"),
    ("V GRD", "ISBL_300"),
    ("V GRD", "ISBL_200"),
    ("V GRD", "HTGL_10"),
    ("U GRD", "ISBL_1000"),
    ("U GRD", "ISBL_975"),
    ("U GRD", "ISBL_950"),
    ("U GRD", "ISBL_900"),
    ("U GRD", "ISBL_850"),
    ("U GRD", "ISBL_800"),
    ("U GRD", "ISBL_700"),
    ("U GRD", "ISBL_600"),
    ("U GRD", "ISBL_500"),
    ("U GRD", "ISBL_400"),
    ("U GRD", "ISBL_300"),
    ("U GRD", "ISBL_200"),
    ("U GRD", "HTGL_10"),
    ("GUST", "SFC_0"),
    ("TMP", "ISBL_1000"),
    ("TMP", "ISBL_975"),
    ("TMP", "ISBL_950"),
    ("TMP", "ISBL_900"),
    ("TMP", "ISBL_850"),
    ("TMP", "ISBL_800"),
    ("TMP", "ISBL_700"),
    ("TMP", "ISBL_600"),
    ("TMP", "ISBL_500"),
    ("TMP", "ISBL_400"),
    ("TMP", "ISBL_300"),
    ("TMP", "ISBL_200"),
    ("TMP", "HTGL_2"),
    ("CAPE", "SFC_0"),
    ("CAPE", "SPDL_0-180"),
    ("CAPE", "SPDL_0-255"),
    ("LFT X", "SFC_0"),
    ("DPT", "HTGL_2"),
    ("CIN", "SFC_0"),
    ("CIN", "SPDL_0-180"),
    ("CIN", "SPDL_0-255"),
    ("P WAT", "EATM_0"),
    ("POT", "SIGL_0.995"),
    ("R H", "ISBL_1000"),
    ("R H", "ISBL_975"),
    ("R H", "ISBL_950"),
    ("R H", "ISBL_900"),
    ("R H", "ISBL_850"),
    ("R H", "ISBL_800"),
    ("R H", "ISBL_700"),
    ("R H", "ISBL_600"),
    ("R H", "ISBL_500"),
    ("R H", "ISBL_400"),
    ("R H", "ISBL_300"),
    ("R H", "HTGL_2"),
    ("V VEL", "ISBL_1000"),
    ("V VEL", "ISBL_975"),
    ("V VEL", "ISBL_950"),
    ("V VEL", "ISBL_900"),
    ("V VEL", "ISBL_850"),
    ("V VEL", "ISBL_800"),
    ("V VEL", "ISBL_700"),
    ("V VEL", "ISBL_600"),
    ("V VEL", "ISBL_500"),
    ("V VEL", "ISBL_400"),
    ("V VEL", "ISBL_300"),
    ("T CDC", "LCY_0"),
    ("T CDC", "MCY_0"),
    ("T CDC", "HCY_0"),
    ("PRATE", "SFC_0"),
    ("PRES", "SFC_0"),
    ("HLCY", "HTGL_0-3000"),
    ("HGT", "ISBL_850"),
    ("HGT", "ISBL_700"),
    ("HGT", "ISBL_500"),
    ("HGT", "ISBL_300"),
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
