import matplotlib.pyplot as plt
import numpy as np

def plot_CO(data, quantities=['CO_QCLS', 'CO_NOAA']):
    at_data = data[data['TYPE'] == 'ATom1']
    days = np.unique(at_data['YYYYMMDD'])

    fig, ax = plt.subplots(figsize=(15, 3*len(days)))
    fig.subplots_adjust(hspace=0.5)
    for j, day in enumerate(days):
        plt.subplot(len(days), 1, j+1)
        ax = plt.gca()
        ax.set_title(day)
        for i, quant in enumerate(quantities):
            day_quant = at_data[(at_data['YYYYMMDD'] == day) & (at_data[quant] != 0)]
            ax.scatter(day_quant['HHMM'], day_quant[quant], 
                       s=10, alpha=0.1,
                       label=quant)
        ax.legend(loc='upper left')

    return fig, ax

