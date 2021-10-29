# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:49:31 2021

@author: ab6801
"""

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Dot, Add, Activation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy as sp


import xarray as xr
xr.set_options(display_style='text')

import xskillscore as xs



from scripts import skill_by_year, add_year_week_coords

from helper_ml_data import load_data, get_basis, clean_coords
from helper_ml_data import preprocess_input, rm_annualcycle, rm_tercile_edges, rm_tercile_edges1, DataGenerator1
from helper_ml_data import pad_earth, add_valid_time_single, single_prediction, skill_by_year_single,slide_predict, add_coords


import warnings
warnings.simplefilter("ignore")
#%%

### set seed to make shuffling in datagenerator reproducible
np.random.seed(42)

# ### define some vars    
v='t2m'

# 2 bi-weekly `lead_time`: week 3-4
lead_input = 0#hind_2000_2019.lead_time[0]
lead_output = 0#obs_2000_2019.lead_time[0]

#%%
#params:
    
smoothing = 0.6
output_dims = 8
input_dims = 32
basis_rad = 16
n_bins = 3


# hyper-parameters for CNN
dropout_rate = 0.4
hidden_nodes = [10] #they tried different architectures
bs=32
ep  = 20

sigma = 9# guassian filtering: best sigma for years 2017/2018


#train domain:
input_lat = slice(81,34)#32
input_lon = slice(49,97)#32

output_lat = slice(63,52)#8          #slice(69,46)#16
output_lon = slice(67,78)#8          #slice(61,85) #16
    
#%%
#read data

path_data = 'local_n'#'server'#
#['../../../../Data/s2s_ai/data', '../../../../Data/s2s_ai']


# ## Hindcast
hind_2000_2019 = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)#['tp','t2m','sm20','sst'])#,'lsm', 'msl', var_list = ['tp','t2m'])#path_data)
hind_2000_2019 = clean_coords(hind_2000_2019)
hind_2000_2019 = hind_2000_2019[['t2m']]

# ## Observations corresponding to hindcasts
obs_2000_2019 = load_data(data = 'obs_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)
# terciled
obs_2000_2019_terciled = load_data(data = 'obs_terciled_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)

#hind_2000_2019_raw = hind_2000_2019_raw[['t2m','t2m_2']]
hind_2000_2019.assign_coords(lead_time = obs_2000_2019.lead_time)##overwrite lead_time... not ideal but ow strange issues..

#mask: same missing values at all forecast_times, masking: only used for label data
mask = xr.where(obs_2000_2019.notnull(),1,np.nan).mean('forecast_time', skipna = False)

#%%

hind_2000_2019[v].isel(forecast_time = 0).mean('realization').plot(cbar_kwargs = {'label': 't2m [K]'})
plt.hlines(y = [34.5, 81], xmin=49.5, xmax = 96 , colors = 'red')#81
plt.vlines(x = [49.5, 96], ymin=34.5, ymax = 81 , colors = 'red')
plt.hlines(y = [52.5, 63], xmin=67.5, xmax = 78 , colors = 'orange')#81
plt.vlines(x = [67.5, 78], ymin=52.5, ymax = 63 , colors = 'orange')
plt.title('Training domains: input (red), output (orange)')

plt.tight_layout()
plt.savefig('plots/train_domain.png')

#%%
#from tensorflow.keras.utils import vis_utils#.plot_model #as vis #import plot_model

#plot model flow chart
#cnn_restored = keras.models.load_model("CNN_patch_34_t2m_20_23")

#tf.keras.utils.plot_model(cnn_restored, to_file='model_plot.png')
#vis.plot_model(cnn_restored)

#%%

fct_train = preprocess_input(hind_2000_2019, v, path_data, lead_input)

fct_train = fct_train.sel(latitude = input_lat, longitude = input_lon)

fct_train.lower_t2m.isel(forecast_time = 0).plot()
fct_train.upper_t2m.isel(forecast_time = 0).plot()
#%%

obs_2000_2019_terciled.sel(latitude = output_lat, longitude = output_lon).t2m.isel(forecast_time = 0).plot(col = 'category')
#%%

data = obs_2000_2019.t2m.isel(latitude = 40, longitude = 40).values

#sort data
x = np.sort(data)

#calculate CDF values
y = 1. * np.arange(len(data)) / (len(data) - 1)

#plot CDF
plt.plot(x, y)

#%%


da = xr.open_dataset('../submissions/24_10_4_withallseeds2/global_prediction_t2m_lead34_smoothed_allyears.nc')['t2m']
da.isel(forecast_time = 0, category = 0).plot()

da = xr.open_dataset('../submissions/24_10_4_withallseeds2/global_prediction_t2m_lead34_raw_allyears.nc')['t2m']
da.isel(forecast_time = 0, category = 0).plot()