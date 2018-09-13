import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import basic_funcs as bf
import datetime as dt

def label_vprofs(data, rolling_avg_window=30, threshold=15):
    data['LAB'] = 0
    days = data['YYYYMMDD'].unique()
    label = 0
    for i, day in enumerate(days):
        day_data = data[data['YYYYMMDD'] == day].reset_index(drop=True).copy()
        day_data, label = label_vprofs_day(day_data, 
                                          init_label=(label+1), 
                                          rolling_avg_window=rolling_avg_window,
                                          threshold=threshold)
        data.loc[data['YYYYMMDD'] == day, 'LAB'] = day_data['LAB'].values
    return data

def label_vprofs_day(data, init_label, rolling_avg_window, threshold):
    # Initialize the label counter
    label = init_label

    # Smooth the data
    mov_avg = data['PRESS'].rolling(rolling_avg_window).mean()
    
    # Find the difference between data points (ie positive or negative slope)
    diff_mov_avg = np.diff(mov_avg)

    # If the slope is always positive or negative within the threshold range,
    # assign a new label. Otherwise, set the label counter equal to the next 
    # largest value.
    for j, item in enumerate(diff_mov_avg):
        if (np.all(diff_mov_avg[j:j+threshold] > 0) | np.all(diff_mov_avg[j:j+threshold] < 0)):
            data.loc[j, 'LAB'] = label
        else:
            if data['LAB'].max() == 0:
                label = init_label
            else:
                label = (data['LAB'].max() + 1).astype(int)
    
    final_label = label
    return data, final_label

def reset_vprof_count(data, label_column='LAB'):
    labels = pd.DataFrame(data[label_column].unique(), columns=[label_column]).reset_index()
    updated_data = pd.merge(left=data, right=labels,
                            how='left',
                            on=label_column)
    updated_data = updated_data.drop(columns=[label_column])
    updated_data = updated_data.rename(columns={'index' : label_column})
    return updated_data

def remove_segmented_profs(data, label_column='LAB', threshold=10):
    labeled_data = data[data['LAB'] > 0].reset_index(drop=True)
    unique_labels, label_counts = np.unique(labeled_data['LAB'], return_counts=True)
    drop_labels = unique_labels[label_counts <= threshold]
    data.loc[data[label_column].isin(drop_labels), label_column] = 0
    return data

def remove_takeoff_landing(data, label_column='LAB', time_threshold_hr=3):
    # min_label = np.min(data[data[label_column] > 0][label_column])
    # max_label = np.max(data[label_column])
    # labels = [min_label, max_label]
    labels = []
    time_threshold = time_threshold_hr*60*60 # time threshold in seconds

    # Create a cumulative hh:mm column
    tot_data = data.copy()
    hhmm_running = pd.to_timedelta(tot_data['HHMM']//100, unit='h') + pd.to_timedelta(tot_data['HHMM']%100, unit='m')
    tot_data['HHMM_running'] = pd.to_datetime(tot_data['YYYYMMDD']) + hhmm_running 
    
    # Find take-offs and landings  
    time_diff = tot_data['HHMM_running'].iloc[1:].reset_index(drop=True)-tot_data['HHMM_running'].iloc[:-1].reset_index(drop=True)
    time_diff = time_diff.dt.total_seconds()
    time_diff = np.append(time_diff, 0)
    idx = np.where((time_diff > time_threshold) | (time_diff < 0))[0]
    for j, index in enumerate(idx):
        landing_label, takeoff_label = bf.nearest_labels(tot_data['LAB'], index)
        labels = labels + [landing_label, takeoff_label]

    # Set label equal to 0
    tot_data.loc[tot_data[label_column].isin(labels), label_column] = 0
    return tot_data

def plot_profiles(labeled_data, label_column='LAB'):
    days = labeled_data['YYYYMMDD'].unique()

    fig, ax = plt.subplots(figsize=(15,1.5*len(days)))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    for j, day in enumerate(days):
        data = labeled_data[labeled_data['YYYYMMDD'] == day]

        plt.subplot(len(days),2,1+j)
        axis = plt.gca()
        axis.set_title(day)
        fig, axis = plot_profiles_day(data, label_column, fig, axis)

    return fig, ax

def plot_profiles_day(data, label_column, fig, ax):
    my_cmap = plt.cm.get_cmap('jet', lut=3000)
    my_cmap._init()
    my_cmap.set_under('gray', alpha=1)

    plot = ax.scatter(data['HHMM'], data['PRESS'], 
                      c=data[label_column], cmap=my_cmap, 
                      vmin=data[label_column].loc[data[label_column]>0].min(), 
                      vmax=data[label_column].loc[data[label_column]>0].max(),
                      s=5)
    fig.colorbar(plot, ax=ax)
    ax.set_yscale('log')
    ax.set_ylim(1000, 200)

    return fig, ax