import numpy as np
import pandas as pd 
import scipy.stats as stats
import sklearn.linear_model as skl
import sklearn.metrics as metrics
import statsmodels.tsa.filters.filtertools as filters
import matplotlib.pyplot as plt
import basic_funcs as bf

MODEL_RUNS_LONG = {'m225' : 'MERRA2, 2x2.5',
                   'm45' : 'MERRA2, 4x5',
                   'g45' : 'GEOSFP, 4x5'}
# DIFF_COLS =  ['M225_OBS', 'M45_OBS', 'G45_OBS', 'M225_M45', 'G45_M45']
DIFF_COLS =  ['OBS', 'OBS_AVG', 'M45', 'M225', 'M45_OBS', 'M225_OBS', 'M225_M45']

def build_summary_df(model_run_dictionary):
    # Make this smarter, eventually.
    if 'LAB' in model_run_dictionary['M225']:
        join_criteria = ['LAB']
        final_cols = ['LAB', 'YYYYMMDD', 'HHMM', 'ATL', 
                      'LAT', 'LON',
                      'M225_OBS', 'M45_OBS', 'G45_OBS']
    else:
        join_criteria = ['YYYYMMDD', 'POINT']
        final_cols = ['TYPE', 'POINT', 
                      'YYYYMMDD', 'HHMM', 'SEASON', 
                      'LAT', 'LON',
                      'M225_OBS', 'M45_OBS', 'G45_OBS']

    df = model_run_dictionary['M225'].drop(columns=['OBS', 'OBS_AVG', 'MOD'])
    df = df.rename(columns={'DIFF' : 'M225_OBS'})

    df = pd.merge(left=df, right=model_run_dictionary['M45'][join_criteria + ['DIFF']],
                  how='left',
                  on=join_criteria)
    df = df.rename(columns={'DIFF' : 'M45_OBS'})

    df = pd.merge(left=df, right=model_run_dictionary['G45'][join_criteria + ['DIFF']],
                  how='left',
                  on=join_criteria)
    df = df.rename(columns={'DIFF' : 'G45_OBS'})

    df = df[final_cols]

    df['M225_M45'] = df['M225_OBS'] - df['M45_OBS']
    df['G45_M45'] = df['G45_OBS'] - df['M45_OBS']

    return df

def build_summary_df_hippo(model_run_cols_dictionary):
    join_criteria = ['LAB']
    final_cols = ['TYPE', 'LAB', 'YYYYMMDD', 'HHMM', 
                  'LAT', 'LON',
                  'OBS', 'OBS_AVG', 'M225', 'M45',
                  'M225_OBS', 'M45_OBS']

    df = model_run_cols_dictionary['M225']
    df = df.rename(columns={'DIFF' : 'M225_OBS',
                            'MOD' : 'M225'})

    df = pd.merge(left=df, right=model_run_cols_dictionary['M45'][join_criteria + ['DIFF', 'MOD']],
                  how='left',
                  on=join_criteria)
    df = df.rename(columns={'DIFF' : 'M45_OBS',
                            'MOD' : 'M45'})

    df = df[final_cols]

    df['M225_M45'] = df['M225_OBS'] - df['M45_OBS']

    return df

def lin_fit(x, y):
    ols_model = skl.LinearRegression(fit_intercept=True)
    ols_model.fit(x.values.reshape(-1, 1),
                  y.values.reshape(-1, 1))
    slope = ols_model.coef_[0]
    intercept = ols_model.intercept_
    pearsonr = stats.pearsonr(x, y)[0]
    return slope, intercept, pearsonr

def obs_mod_scatter(data, axis=None):
    if axis is None:
        fig, ax = plt.subplots(figsize=(3,3))

    xymin = data['OBS'].values.min()-50
    xymax = data['OBS'].values.max()+50
 
    slope, intercept, pearsonr = lin_fit(data['OBS'], data['MOD'])
    
    x_fit = np.array((xymin, xymax))
    y_fit = intercept+slope*x_fit
    
    axis.set_facecolor('0.95')
    axis.scatter(data['OBS'], data['MOD'], 
                 color='midnightblue', alpha=0.2,
                 label='Data')
    axis.plot(x_fit, y_fit, 
              '--', color='coral', linewidth=1, 
              label='Best Fit')
    axis.plot(x_fit, x_fit, 
              '-', color='0.65', linewidth=1, 
              label='1-1 Line')
    axis.text(xymin+10, xymax-10,
              'Pearson R = %.3f\ny = %.3f + %.3fx' % (pearsonr, intercept, slope), 
               fontsize=12, verticalalignment='top')
    
    axis.set_xlim(xymin, xymax)
    axis.set_ylim(xymin, xymax)

    axis.set_xlabel('Observed Methane Levels (ppb)')
    axis.set_ylabel('Modeled Methane Levels (ppb)')
    
    axis.legend(loc='lower right')

    return axis

def lat_scatter(data, axis=None):
    if axis is None:
        fig, axis = plt.subplots(figsize=(3,3))

    ymin = data['OBS'].values.min()-50
    ymax = data['OBS'].values.max()+50

    obs_slope, obs_intercept, _ = lin_fit(data['LAT'], data['OBS'])
    mod_slope, mod_intercept, _ = lin_fit(data['LAT'], data['MOD'])

    axis.set_facecolor('0.95')
    axis.scatter(data['LAT'], data['OBS'], 
                 color='midnightblue', alpha=0.2,
                 label='Observations, OLS Slope = %.3f' % obs_slope)
    axis.scatter(data['LAT'], data['MOD'], 
                 color='coral', alpha=0.2,
                 label='Modeled, OLS Slope %.3f' % mod_slope)
    
    axis.set_xlim(-90, 90)
    axis.set_ylim(ymin, ymax)

    axis.set_xlabel('Latitude')
    axis.set_ylabel('Methane Levels (ppb)')
    
    axis.legend(loc='upper left')

    return axis

def lat_diff_scatter(data, axis=None):
    if axis is None:
        fig, axis = plt.subplots(figsize=(3,3))

    diff = data['MOD']-data['OBS']

    absmax = np.max((np.abs(diff.values.min()), diff.values.max()))
    ymin = -absmax - 10
    ymax = absmax + 10

    axis.set_facecolor('0.95')
    axis.scatter(data['LAT'], diff, 
                 color='midnightblue', alpha=0.2)

    axis.set_xlim(-90, 90)
    axis.set_ylim(ymin, ymax)

    axis.set_xlabel('Latitude')
    axis.set_ylabel('Model - Observations')
    
    return axis

def band_scatter(data, axis=None):
    if axis is None:
        fig, axis = plt.subplots(figsize=(6,6))

    diff_cols = {'M45_OBS' : 'MERRA2 4x5 - Observations',
                 'M225_OBS' : 'MERRA2 2x2.5 - Observations',
                 'M225_M45' : 'MERRA2 2x2.5 - MERRA2 4x5', 
                 'G45_M45' : 'GEOSFP 4x5 - MERRA2 4x5'}

    colors = {'M45_OBS' : '0.25',
              'M225_OBS' : '0.75',
              'M225_M45' : 'coral', 
              'G45_M45' : 'midnightblue'}

    linestyles = {'M45_OBS' : '--', 
                  'M225_OBS' : '--',
                  'M225_M45' : '-', 
                  'G45_M45' : '-'}

    axis.set_facecolor('0.95')
    for quantity, label in diff_cols.items():
        y = data[quantity]
        axis.plot(np.arange(len(y)), y, label=label,
                  c=colors[quantity], linestyle=linestyles[quantity], 
                  marker='.', markersize=10)

    axis.set_xlim(-0.5,len(y)-0.5)
    axis.set_xticks(np.arange(len(y)))
    axis.set_xticklabels(data['BAND'].unique())
    axis.set_xlabel('Bands')

    ymin = np.nanmin(data[DIFF_COLS].values)-5
    ymax = np.nanmax(data[DIFF_COLS].values)+5
    axis.set_ylim(ymin, ymax)

    axis.legend()

    return axis

def plot_atom_bias(atom_data, scatter_function, title_dictionary, sharey=True):
    i = 0
    fig, ax = plt.subplots(2, len(atom_data), figsize=(10, 6))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.075, wspace=0.075)
    for mr, mr_data in atom_data.items():
        ax[0,i].set_title(title_dictionary[mr], fontsize=14)
        ax[0,i] = scatter_function(mr_data[mr_data['ATL']], axis=ax[0,i])
        ax[1,i] = scatter_function(mr_data[~mr_data['ATL']], axis=ax[1,i])
        
        ax[0,i].set_xlabel('')
        plt.setp(ax[0,i].get_xticklabels(), visible=False)
        
        if (i > 0) & sharey:
            for j in range(2):
                ax[j,i].set_ylabel('')
                plt.setp(ax[j,i].get_yticklabels(), visible=False)
        
        i += 1
    
    max_axis = len(atom_data)-1
    ax[0,max_axis].text(ax[0,max_axis].get_xlim()[-1], 
                              np.mean(ax[0,max_axis].get_ylim()), 
                              'Atlantic', 
                              rotation=270, fontsize=14, verticalalignment='center')
    ax[1,max_axis].text(ax[1,max_axis].get_xlim()[-1], 
                              np.mean(ax[1,max_axis].get_ylim()), 
                              'Pacific',
                               rotation=270, fontsize=14, verticalalignment='center')

    return fig, ax

def plot_noaa_bias(noaa_data, scatter_function, title_dictionary, sharey=True):
    i = 0
    fig, ax = plt.subplots(4, len(noaa_data), figsize=(10, 12))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.075, wspace=0.05)
    for mr, mr_data in noaa_data.items():
        if 'REM' in mr_data:
            data = mr_data[mr_data['REM']]
        else:
            data = mr_data.copy()

        seasons = ['DJF', 'MAM', 'JJA', 'SON']

        ax[0,i].set_title(title_dictionary[mr], fontsize=14)
        for j, season in enumerate(seasons): 
            ax[j,i] = scatter_function(data[data['SEASON'] == season], 
                                       axis=ax[j,i])
            
            if (i > 0) & sharey:
                for k in range(4):
                    ax[k,i].set_ylabel('')
                    plt.setp(ax[k,i].get_yticklabels(), visible=False)
            
            if j < 3:
                ax[j,i].set_xlabel('')
                plt.setp(ax[j,i].get_xticklabels(), visible=False)

        i += 1

    max_axis = len(noaa_data)-1
    for j, season in enumerate(seasons):
        ax[j,max_axis].text(ax[j,max_axis].get_xlim()[-1], 
                            np.mean(ax[j,max_axis].get_ylim()),
                            season,
                            rotation=270, fontsize=14, verticalalignment='center')   

    return fig, ax

# def plot_hippo_bias(hippo_data, scatter_function, title_dictionary, sharey=True):


def calc_band_stats(summary_data, grouping_quantity, groupings=None):
    summary = summary_data.copy()
    band_mean = calc_band_stat(summary_data, 'mean')
    band_std = calc_band_stat(summary_data, 'std')
    band_stats = {'mean' : band_mean, 'std' : band_std}

    return band_stats

def calc_band_stat(data, stat, grouping_quantity, groupings=None):
    if groupings is not None:
        data['BAND'] = pd.cut(data[grouping_quantity],
                                 bins=groupings)

    if 'LAB' in data:
        if 'ATL' in data:
            groupby_list = ['TYPE', 'ATL', 'BAND']
        else:
            groupby_list = ['TYPE', 'BAND']
    else:
        groupby_list = ['SEASON', 'BAND']

    if str.lower(stat)=='std':
        band_stat = data.groupby(groupby_list).std()[DIFF_COLS]
    elif str.lower(stat)=='mean':
        band_stat = data.groupby(groupby_list).mean()[DIFF_COLS]

    band_stat = band_stat.reset_index()
    band_stat = band_stat.sort_values(by=groupby_list)

    counts = data.groupby(groupby_list).count()[DIFF_COLS[0]]
    counts = counts.reset_index()
    counts = counts.rename(columns={DIFF_COLS[0] : 'COUNT'})

    band_stat = pd.merge(left=band_stat, right=counts,
                         how='left',
                         on=groupby_list)

    return band_stat