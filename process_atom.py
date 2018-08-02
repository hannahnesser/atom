import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import datetime

FILE_PATH = '/Users/hannahnesser/Documents/Harvard/Research/ATom/'

AT_DATES = [item[-8:] for item in listdir(FILE_PATH+'planeflight_files/ATom')
            if (item[0] != '.')]
# AT_DATES = ['20160729', '20160801', '20160803', '20160806', '20160808',
#            '20160812', '20160815', '20160817', '20160820', '20160822',
#            '20160823']

COL_NAMES = ['Point', 'Type', 'DD-MM-YYYY', 'HH:MM',
             'LAT', 'LON', 'ALT/PRE', 'OBS']

COL_NAMES_GC = ['POINT', 'TYPE', 'YYYYMMDD', 'HHMM', 
                'LAT', 'LON', 'PRESS', 'OBS', 
                'T-IND', 'P-I', 'I-IND', 'J-IND', 
                'TROP', 'TRA_001']

def read_at(dates, filepath):
    # Load data into a single dataframe
    at_files = [filepath + 'ATom_long/atom_' 
                + date + '.csv' for date in dates]
    at_files = np.sort(at_files)
    at_data = pd.DataFrame()
    for i, file in enumerate(at_files):
        data = pd.read_csv(file)
        at_data = pd.concat([at_data, data])

    return at_data

def process_at(dates=AT_DATES, filepath=FILE_PATH):
    at_data = read_at(dates, filepath)
    at_data = at_data.rename(columns={'POINT' : 'AT_POINT',
                                      'DD-MM-YYYY' : 'YYYYMMDD',
                                      'OBS' : 'AT_OBS'})
    
    # Format data to be consistent with planeflight files
    # Dates
    at_data['YYYYMMDD'] = pd.to_datetime(at_data['YYYYMMDD'], format='%d-%m-%Y').dt.date

    # Type
    at_data['TYPE'] = at_data['TYPE'].str.strip()

    # We will handle this in the planeflight module.
    # # Alt/Pre
    # at_data = at_data.loc[at_data['ALT/PRE'] > 0]

    # # Trust the existing points. ## maybe not.
    # # AT Point
    # at_data['AT_POINT'] = 1+np.arange(at_data.shape[0])

    # Subset
    at_data = at_data[['AT_POINT', 'YYYYMMDD', 'AT_OBS', 'CO_NOAA', 'CO_QCLS', 'O3', 'PROF']]
    return at_data

def read_pressure_levels(filepath=FILE_PATH+'pressure_levels.csv'):
    # Read pressure level file 
    pressure_levels = pd.read_csv(filepath)

    # Calculate delta P
    pressure_levels['P-DIFF'] = pressure_levels['P-MIN'] - pressure_levels['P-MAX']

    # Find the pressure centers of the consolidated pressure levels s
    s_cents = pd.DataFrame()
    s_cents['P-I-S'] = np.arange(0,10)
    s_cents['P-MIN-S'] = pressure_levels.groupby(['P-I-S']).max()['P-MIN'].reset_index()['P-MIN']
    s_cents['P-MAX-S'] = pressure_levels.groupby(['P-I-S']).min()['P-MAX'].reset_index()['P-MAX']
    s_cents['P-CENT-S'] = ((s_cents['P-MIN-S'] + s_cents['P-MAX-S'])/2)
    s_cents['P-DIFF-S'] = s_cents['P-MIN-S'] - s_cents['P-MAX-S']
    s_cents = s_cents[['P-I-S', 'P-CENT-S', 'P-DIFF-S']]
    pressure_levels = pd.merge(left=pressure_levels, right=s_cents,
                               how='left',
                               on='P-I-S')

    pressure_centers = pressure_levels[['P-I', 'P-I-S', 'P-CENT', 'P-CENT-S', 'P-DIFF', 'P-DIFF-S']]

    return pressure_levels, pressure_centers

def read_noaa(filepath=FILE_PATH+'NOAA_site_codes.csv'):
    noaa_sites = pd.read_csv('/Users/hannahnesser/Documents/Harvard/Research/ATom/NOAA_site_codes.csv')
    noaa_sites = noaa_sites.rename(columns={'Code' : 'TYPE', 'Remote' : 'REM'})
    noaa_sites['REM'] = noaa_sites['REM'].astype(bool)
    noaa_sites = noaa_sites[['TYPE', 'REM']]
    return noaa_sites

def read_pf(filepath):
    pf_files = [filepath + f for f in listdir(filepath) 
                if (isfile(join(filepath, f))
                    and f[0] != '.')]
    pf_files = np.sort(pf_files)
    
    pf_tot = pd.DataFrame()
    for i, file in enumerate(pf_files):
        data = pd.read_csv(file, usecols = COL_NAMES_GC, sep = '[\s]{1,20}',
                           engine='python')
        data.insert(1, 'AT_POINT', -1)
        data.loc[data['TYPE'].str.slice(0,4)=='ATom', 'AT_POINT'] = 1 + np.arange(data[data['TYPE'].str.slice(0,4)=='ATom'].shape[0])
        pf_tot = pd.concat((pf_tot, data))

    return pf_tot

def process_pf(filepath, at_data, pressure_centers, noaa_sites):
    pf_tot = read_pf(filepath)

    # Change column name to reflect modeled observations
    pf_tot = pf_tot.rename(columns={'TRA_001': 'MOD', 'TROP': 'TROP_MOD'})

    # Scale up ppb to compare to observed quantity
    pf_tot['MOD'] = pf_tot['MOD']*10**9 #ppb

    # Fix data types
    pf_tot['POINT'] = pf_tot['POINT'].astype(int)
    pf_tot['P-I'] = pf_tot['P-I'].astype(int)
    pf_tot['I-IND'] = pf_tot['I-IND'].astype(int)
    pf_tot['J-IND'] = pf_tot['J-IND'].astype(int)

    # # Add a column just for ATom point values
    # pf_tot['AT_POINT'] = -1
    # pf_tot.loc[pf_tot['TYPE'].str.slice(0,4) == 'ATom', 'AT_POINT'] = 1+np.arange(pf_tot[pf_tot['TYPE'].str.slice(0,4)=='ATom'].shape[0])

    # Add season identifier while YYYYMMDD is still a string
    pf_tot = define_seasons(pf_tot)
    
    # Add Atlantic/Pacific identifier
    overland_dates = [datetime.date(2016, 8, 22), datetime.date(2016, 8, 23)]
    pf_tot = define_oceans(pf_tot, arctic_boundary=66, non_ocean_dates=overland_dates)

    # Identify remote vs. non-remote sites in the NOAA surface sites
    pf_tot = pd.merge(left=pf_tot, right=noaa_sites,
                     how='left',
                     on='TYPE')
    pf_tot['REM'] = pf_tot['REM'].astype(bool)
    
    # Bring in CO/O3 data for identification of stratosphere
    pf_tot = pd.merge(left=pf_tot, right=at_data,
                      how='left',
                      on=['TYPE', 'AT_POINT', 'YYYYMMDD'])
    
    # Change GC troposphere identifier to a boolean
    pf_tot.loc[pf_tot['TROP_MOD'] == 'T', 'TROP_MOD'] = True
    pf_tot.loc[pf_tot['TROP_MOD'] == 'F', 'TROP_MOD'] = False

    # Bring in pressure center
    pf_tot = pd.merge(left=pf_tot, right=pressure_centers,
                     how='left',
                     on=['P-I'])

    # Throw out areas where we don't have pressure measurements
    pf_tot = pf_tot.loc[pf_tot['PRESS'] > 0]
    
    print('Processing of', filepath.split('/')[-2], 'complete.')
    return pf_tot

def define_oceans(data, arctic_boundary, non_ocean_dates):
    # Use NOAA's boundaries for the oceans: 
    # https://www.nodc.noaa.gov/woce/woce_v3/wocedata_1/woce-uot/summary/bound.htm
    data['OCEAN'] = 'NONE'
    data.loc[(data['TYPE'].str.slice(0,4) == 'ATom') 
               & (data['LON'] < -70) 
               & (data['LAT'] < arctic_boundary), 
               'OCEAN'] = 'PAC'
    data.loc[(data['TYPE'].str.slice(0,4) == 'ATom') 
               & (data['LON'] > -70) 
               & (data['LAT'] < arctic_boundary), 
               'OCEAN'] = 'ATL'
    data.loc[(data['TYPE'].str.slice(0,4) == 'ATom') 
               & (data['LAT'] > arctic_boundary), 
               'OCEAN'] = 'ARC'
    data.loc[data['YYYYMMDD'].isin(non_ocean_dates), 'OCEAN'] = 'NONE'
    return data

def define_seasons(data):
    data['YYYYMMDD'] = pd.to_datetime(data['YYYYMMDD'], format='%Y%m%d')
    data['SEASON'] = 'DJF'
    data.loc[data['YYYYMMDD'].dt.month.isin([3,4,5]), 'SEASON'] = 'MAM'
    data.loc[data['YYYYMMDD'].dt.month.isin([6,7,8]), 'SEASON'] = 'JJA'
    data.loc[data['YYYYMMDD'].dt.month.isin([9,10,11]), 'SEASON'] = 'SON'
    data['YYYYMMDD'] = data['YYYYMMDD'].dt.date
    return data