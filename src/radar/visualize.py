from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from wind_forecast.loaders.CMAXLoader import CMAXLoader

if __name__ == "__main__":
    date = datetime(2017, 8, 11, 20)
    cmax_loader = CMAXLoader()
    cmax_image = cmax_loader.get_cmax_image(CMAXLoader.get_date_key(date))
    sns.heatmap(cmax_image)
    plt.show()
