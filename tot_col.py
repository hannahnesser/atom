import vert_profs as vp
import pandas as pd
import numpy as np

## Rewrite calc_tot_cols yikes.

def calc_tot_cols(full_data, pressure_bands, pressure_diff=None, label_column='LAB', calc_partial_col=False):
    if type(pressure_bands) == str:
        full_data = full_data.rename(columns={pressure_bands : 'P_BANDS',
                                              pressure_diff : 'P_DIFF'})
        pressure_bands = np.unique(data[pressure_bands])
    else:
        if type(pressure_bands) == int:
            pressure_bands = np.linspace(325, hippo_tot['M225']['PRESS'].max(), 6)
        full_data['P_BANDS'] = pd.cut(full_data['PRESS'], bins=pressure_bands)
        right = full_data['P_BANDS'].apply(lambda x: x.right).astype(float)
        left = full_data['P_BANDS'].apply(lambda x: x.left).astype(float)
        full_data['P_DIFF'] = right - left
    
    column_list = [label_column, 'P_BANDS', 'P_DIFF', 'TYPE', 'YYYYMMDD', 'HHMM',
                   'LAT', 'LON', 'OBS', 'OBS_AVG', 'MOD', 'DIFF']
    columns_datetime = ['TYPE', label_column, 'YYYYMMDD', 'HHMM']
    if 'OCEAN' in full_data:
        column_list = column_list + ['OCEAN']
        columns_datetime = columns_datetime + ['OCEAN']
        
    data = full_data[column_list]
    
    data = complete_vprofs(data, threshold=3, pressure_levels=(len(pressure_bands)-1))
    # data = vp.reset_vprof_count(data, label_column=label_column)

    # Create the basis of the total column dataframe
    tot_cols = data[[label_column, 'LAT', 'LON']].groupby([label_column]).mean().reset_index()
    date_time = data[columns_datetime].groupby([label_column]).min().reset_index()
    tot_cols = pd.merge(left=date_time, right=tot_cols,
                        how='right',
                        on=label_column)

    # Calculate columns
    pavg_data = data[[label_column, 'P_BANDS', 'P_DIFF', 'OBS', 'OBS_AVG', 'MOD', 'DIFF']]
    pavg = pavg_data.groupby([label_column, 'P_BANDS']).mean().reset_index()
    for i, col_title in enumerate(['OBS', 'OBS_AVG', 'MOD', 'DIFF']):
        pavg[col_title] = pavg[['P_DIFF']].values*pavg[[col_title]]
        
    sums = pavg.groupby([label_column]).sum().reset_index()

    for i, col_title in enumerate(['OBS', 'OBS_AVG', 'MOD', 'DIFF']):
        sums[col_title] = sums[[col_title]]/sums[['P_DIFF']].values
        
    sums = sums[[label_column, 'OBS', 'OBS_AVG', 'MOD', 'DIFF']]
    
    # Combine
    tot_cols = pd.merge(left=tot_cols, right=sums,
                        how='left',
                        on=label_column)

    return tot_cols

# def calc_partial_cols(pavg_data, label_column, pressure_grouping):
#     # P_BAND 1-4 take us down to ~500 hPa.
#     pavg_data['P_BAND'] = pd.cut(pavg_data['P_BAND'], 
#                                  bins=[0,4,6], 
#                                  labels=['Surface - 490 hPa', '490 hPa - 288 hPa'])
#     sums = pavg_data.groupby(['LAB', 'P_BAND']).sum().reset_index()
#     return sums

def complete_vprofs(full_data, label_column='LAB', threshold=3, pressure_levels=6):
    data = full_data[(full_data[label_column] > 0) & (~full_data['P_BANDS'].isna())]

    # we define a complete vertical profile as one which has at least
    # n=threshold datapoints in each of the reduced pressure levels
    data_counts = data.groupby([label_column, 'P_BANDS']).count()['YYYYMMDD'].reset_index()
    data_counts = data_counts.rename(columns={'YYYYMMDD' : 'COUNT'})
    data_counts = data_counts[data_counts['COUNT'] > threshold]

    plev_counts = data_counts.groupby([label_column]).count()['COUNT'].reset_index()
    plev_counts = plev_counts[plev_counts['COUNT'] == pressure_levels]
    
    data = data[data[label_column].isin(plev_counts[label_column])]
    return data