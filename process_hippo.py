import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import datetime

FILE_PATH = '/Users/hannahnesser/Documents/Harvard/Research/HIPPO/'

COL_NAMES = ['Point', 'Type', 'DD-MM-YYYY', 'HH:MM',
             'LAT', 'LON', 'ALT/PRE', 'OBS']

COL_NAMES_GC = ['POINT', 'TYPE', 'YYYYMMDD', 'HHMM', 
                'LAT', 'LON', 'PRESS', 'OBS', 
                'T-IND', 'P-I', 'I-IND', 'J-IND', 
                'TRA_001']

HIPPO_DATES = [20090109, 20090110, 20090112, 20090113, 20090114, 20090115,
               20090116, 20090117, 20090118, 20090119, 20090120, 20090121,
               20090123, 20090124, 20090127, 20090128, 20090129, 20090130,
               20091020, 20091022, 20091031, 20091102, 20091103, 20091104,
               20091105, 20091107, 20091108, 20091109, 20091110, 20091111,
               20091112, 20091114, 20091115, 20091116, 20091117, 20091119,
               20091120, 20091121, 20091122, 20100316, 20100318, 20100324,
               20100325, 20100326, 20100329, 20100330, 20100331, 20100401,
               20100402, 20100403, 20100405, 20100406, 20100408, 20100409,
               20100410, 20100411, 20100413, 20100414, 20100415, 20100416,
               20110607, 20110609, 20110614, 20110616, 20110617, 20110618,
               20110619, 20110622, 20110623, 20110625, 20110626, 20110628,
               20110629, 20110701, 20110703, 20110706, 20110707, 20110710,
               20110711, 20110809, 20110811, 20110816, 20110818, 20110819,
               20110820, 20110822, 20110823, 20110824, 20110825, 20110827,
               20110828, 20110829, 20110830, 20110901, 20110902, 20110903,
               20110904, 20110906, 20110907, 20110908, 20110909]

def read_pf(filepath):
    pf_files = [filepath + f for f in listdir(filepath) 
                if isfile(join(filepath, f))
                and (f[0] != '.')
                and (int(f[-8:]) in (HIPPO_DATES))]
    pf_files = np.sort(pf_files)
    
    pf_tot = pd.DataFrame(columns=COL_NAMES_GC)
    for i, file in enumerate(pf_files):
        data = pd.read_csv(file, usecols = COL_NAMES_GC, sep = '[\s]{1,20}',
                           engine='python')
        pf_tot = pd.concat((pf_tot, data))

    return pf_tot

def process_pf(filepath):
    pf_tot = read_pf(filepath)

    # Get rid of NOAA observations
    pf_tot = pf_tot[pf_tot['TYPE'].str.slice(0,4) == 'hipp']

    # Throw out areas where we don't have pressure measurements
    pf_tot = pf_tot.loc[pf_tot['PRESS'] > 0]

    # Get rid of 0 observations
    pf_tot = pf_tot[pf_tot['OBS'] > 0]

    # Change column name to reflect modeled observations
    pf_tot = pf_tot.rename(columns={'TRA_001': 'MOD'})

    # Scale up ppb to compare to observed quantity
    pf_tot['MOD'] = pf_tot['MOD']*10**9 #ppb 

    # Fix data types
    pf_tot['POINT'] = pf_tot['POINT'].astype(int)
    pf_tot['P-I'] = pf_tot['P-I'].astype(int)
    pf_tot['I-IND'] = pf_tot['I-IND'].astype(int)
    pf_tot['J-IND'] = pf_tot['J-IND'].astype(int)
    
    print('Processing of', filepath.split('/')[-2], 'complete.')
    return pf_tot