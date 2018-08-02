import pandas as pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def curtain(data, quantity, absolute=True):
    if absolute:
        cmap = plt.cm.get_cmap('jet', lut=3000)
        vmin = 1700
        vmax = 2000
    else:
        cmap = plt.cm.get_cmap('bwr', lut=3000)
        vmin = -100
        vmax = 100

    hippo_flights = np.unique(data['TYPE'])

    fig, ax = plt.subplots(1,len(hippo_flights),figsize=(8*len(hippo_flights),7),sharey=True)
    plt.tight_layout()
    cmap = plt.cm.get_cmap('jet', lut=3000)

    for i, hippo in enumerate(hippo_flights):
        data_hippo = data[data['TYPE'] == hippo]
        axis = ax[i].scatter(data_hippo['LAT'], data_hippo['PRESS'], c=data_hippo[quantity],
                             cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i].set_xlabel('Latitude', fontsize=16)
        ax[i].set_xlim(-90, 90)
        ax[i].set_yscale('log')
        ax[i].set_ylim(data['PRESS'].max(), data['PRESS'].min())
        ax[i].set_title(hippo)

    ax[0].set_ylabel('Pressure', fontsize=16)
    cax = fig.add_axes([1, 0.07, 0.02, 0.885])
    plt.colorbar(axis, cax=cax)

    return fig, ax