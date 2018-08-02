import numpy as np
import pandas as pd

def calc_bias(data, quantity1, quantity2):
    bias_data = data[(data[quantity1] > 0) & (data[quantity2] > 0)]
    return np.mean(bias_data[quantity1] - bias_data[quantity2])

def avg_gc_grid(data, avg_quantity):
    if 'TROP' in data:
        sub_data = data.loc[(data['TROP_'+avg_quantity] & (data[avg_quantity] != 0))]
    else:
        sub_data = data.loc[data[avg_quantity] != 0]

    at_avg = sub_data.groupby(['TYPE', 'P-I', 'I-IND', 'J-IND'])[avg_quantity].mean()
    at_avg = at_avg.reset_index()
    at_avg = at_avg.rename(columns={avg_quantity : (avg_quantity+'_AVG')})
    data = pd.merge(left=data, right=at_avg,
                    how='left',
                    on=['TYPE', 'P-I', 'I-IND', 'J-IND'])
    return data

def nearest_labels(labels, idx):
    fwd_labels = labels[idx+1:].reset_index(drop=True)
    fwd_label = first_non_zero(fwd_labels)

    bwd_labels = np.flip(labels[:idx], axis=0).reset_index(drop=True)
    bwd_label = first_non_zero(bwd_labels)

    return bwd_label, fwd_label

def first_non_zero(labels):
    i = np.where(labels > 0)[0][0]
    return labels[i]