import pandas as pandas
import matplotlib
import matplotlib.pyplot as plt

ocean_key = {'ATL' : 'Atlantic Ocean', 'PAC' : 'Pacific Ocean', 'ARC' : 'Arctic Ocean'}

def basic_curtain(data, quantity, oceans=['ATL', 'PAC', 'ARC']):
    fig, ax = plt.subplots(1,len(oceans),figsize=(8*len(oceans),7),sharey=True)
    plt.tight_layout()
    cmap = plt.cm.get_cmap('jet', lut=3000)
    for i, ocean in enumerate(oceans):
        ocean_data = data[data['OCEAN'] == ocean]
        axis = ax[i].scatter(ocean_data['LAT'], ocean_data['PRESS'], c=ocean_data[quantity], 
                             cmap=cmap)
        ax[i].set_title(ocean_key[ocean], fontsize=18)
    
    ax[0].set_ylabel('Pressure', fontsize=16)
    cax = fig.add_axes([1, 0.07, 0.02, 0.885])
    plt.colorbar(axis, cax=cax)

    for i in range(len(oceans)):
        ax[i].set_yscale('log')
        ax[i].set_ylim(data['PRESS'].max(), data['PRESS'].min())
        ax[i].set_xlim(-85, 85)
        ax[i].set_xlabel('Latitude', fontsize=16)

    ax[(len(oceans)-1)].set_xlim(60,90)

    return ax

def absolute_curtain(data, quantity, oceans=['ATL', 'PAC', 'ARC']):
    fig, ax = plt.subplots(1,len(oceans),figsize=(8*len(oceans),7),sharey=True)
    plt.tight_layout()
    cmap = plt.cm.get_cmap('jet', lut=3000)
    for i, ocean in enumerate(oceans):
        ocean_data = data[data['OCEAN'] == ocean]
        axis = ax[i].scatter(ocean_data['LAT'], ocean_data['PRESS'], c=ocean_data[quantity], 
                             cmap=cmap, vmin=1700, vmax=2000)
        ax[i].set_title(ocean_key[ocean], fontsize=18)
    
    ax[0].set_ylabel('Pressure', fontsize=16)
    cax = fig.add_axes([1, 0.07, 0.02, 0.885])
    plt.colorbar(axis, cax=cax)

    for i in range(len(oceans)):
        ax[i].set_yscale('log')
        ax[i].set_ylim(data['PRESS'].max(), data['PRESS'].min())
        ax[i].set_xlim(-85, 85)
        ax[i].set_xlabel('Latitude', fontsize=16)

    ax[(len(oceans)-1)].set_xlim(60,90)

    return ax

def relative_curtain(data, quantity):
    data_atl = data[data['ATL']]
    data_pac = data[~data['ATL']]
    fig, ax = plt.subplots(1,2,figsize=(16,7),sharey=True)
    plt.tight_layout()
    cmap = plt.cm.get_cmap('bwr', lut=3000)
    ax[0].scatter(data_atl['LAT'], data_atl['PRESS'], c=data_atl[quantity], 
                  cmap=cmap, vmin=-100, vmax=100)
    ax[0].set_ylabel('Pressure', fontsize=16)
    ax[0].set_title('Atlantic Ocean', fontsize=18)
    ax1 = ax[1].scatter(data_pac['LAT'], data_pac['PRESS'], c=data_pac[quantity], 
                  cmap=cmap, vmin=-100, vmax=100)
    ax[1].set_title('Pacific Ocean', fontsize=18)
    cax = fig.add_axes([1, 0.07, 0.02, 0.885])
    plt.colorbar(ax1, cax=cax)

    for i in range(2):
        ax[i].set_facecolor('0.97')
        ax[i].set_yscale('log')
        ax[i].set_ylim(data['PRESS'].max(), data['PRESS'].min())
        ax[i].set_xlim(-85, 85)
        ax[i].set_xlabel('Latitude', fontsize=16)

    return fig, ax