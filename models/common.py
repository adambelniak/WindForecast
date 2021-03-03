import matplotlib.pyplot as plt
import numpy as np

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