import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.lines import Line2D

import cartopy.crs as ccrs

def map(data, quantity):
    plt.figure(figsize=(12,3))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
    axis = ax.scatter(data['LON'], data['LAT'], c=data[quantity])
    
    cb = plt.colorbar(axis)

    return ax, cb

def heat_map(data, quantity, quantity_name, absolute=True, vrange=None, central_longitude=0):
    plt.figure(figsize=(20,5))
    if absolute:
        cmap = plt.cm.get_cmap('jet')
        if vrange is None:
            vmin = 1700
            vmax = 1950
    else:
        cmap = plt.cm.get_cmap('bwr')
        if vrange is None:
            vmin = -max(abs(data[quantity]))
            vmax = -vmin

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax = plt.axes(projection=proj)
    ax.coastlines()
    # ax.gridlines(draw_labels=True)
    # ax.set_xlim(-180, 180)
    # ax.set_ylim(-90, 90)

    axis = ax.scatter(data['LON']-central_longitude, data['LAT'],
                      c=data[quantity],
                      cmap=cmap,
                      s=90,
                      vmin=vmin, vmax=vmax)

    cb = plt.colorbar(axis)
    cb.ax.set_ylabel(quantity_name)

    return ax