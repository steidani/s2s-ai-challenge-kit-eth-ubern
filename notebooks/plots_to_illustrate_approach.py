# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:49:31 2021

@author: ab6801
"""
from tensorflow.keras.layers import Reshape

import matplotlib.pyplot as plt
import numpy as np

import xarray as xr
xr.set_options(display_style='text')

from helper_ml_data import load_data, clean_coords, preprocess_input

import warnings
warnings.simplefilter("ignore")
#%%

# ### define some vars    
v='t2m'

# 2 bi-weekly `lead_time`: week 3-4
lead_input = 0#hind_2000_2019.lead_time[0]
lead_output = 0#obs_2000_2019.lead_time[0]

#%%
#params:

sigma = 9# guassian filtering: best sigma for years 2017/2018


#train domain:
input_lat = slice(81,34)#32
input_lon = slice(49,97)#32

output_lat = slice(63,52)#8          #slice(69,46)#16
output_lon = slice(67,78)#8          #slice(61,85) #16
    
#%%
#read data

path_data = 'local_n'#'server'#['../../../../Data/s2s_ai/data', '../../../../Data/s2s_ai']


# ## Hindcast
hind_2000_2019 = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)
hind_2000_2019 = clean_coords(hind_2000_2019)
hind_2000_2019 = hind_2000_2019[['t2m']]

# ## Observations corresponding to hindcasts
obs_2000_2019 = load_data(data = 'obs_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)
# terciled
obs_2000_2019_terciled = load_data(data = 'obs_terciled_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)

#hind_2000_2019_raw = hind_2000_2019_raw[['t2m','t2m_2']]
#hind_2000_2019.assign_coords(lead_time = obs_2000_2019.lead_time)##overwrite lead_time... not ideal but ow strange issues..

#mask: same missing values at all forecast_times, masking: only used for label data
#mask = xr.where(obs_2000_2019.notnull(),1,np.nan).mean('forecast_time', skipna = False)

#%%

#plot training domain
plt.figure()
hind_2000_2019[v].isel(forecast_time = 0).mean('realization').plot(cbar_kwargs = {'label': 't2m [K]'})
plt.hlines(y = [34.5, 81], xmin=49.5, xmax = 96 , colors = 'red', alpha = 0.6)#81
plt.vlines(x = [49.5, 96], ymin=34.5, ymax = 81 , colors = 'red', alpha = 0.6)
plt.hlines(y = [52.5, 63], xmin=67.5, xmax = 78 , colors = 'orange', alpha = 0.6)#81
plt.vlines(x = [67.5, 78], ymin=52.5, ymax = 63 , colors = 'orange', alpha = 0.6)
#shifted
plt.hlines(y = [34.5, 81], xmin=60, xmax = 106.5, colors = 'red', linestyles = 'dashed')#81
plt.vlines(x = [60, 106.5], ymin=34.5, ymax = 81 , colors = 'red', linestyles = 'dashed')
plt.hlines(y = [52.5, 63], xmin=78, xmax = 88.5 , colors = 'orange', linestyles = 'dashed')#81
plt.vlines(x = [78, 88.5], ymin=52.5, ymax = 63 , colors = 'orange', linestyles = 'dashed')
#plt.title('Training domains: input (red), output (orange)')
plt.title('')
plt.tight_layout()
plt.savefig('plots/train_domain.png', dpi = 1200)

#%%


#plot model flow chart
#cnn_restored = keras.models.load_model("CNN_patch_34_t2m_20_23")
#import tensorflow as tf
#tf.keras.utils.plot_model(cnn_restored, to_file='model_plot.png')
#from tensorflow.keras.utils import vis_utils
#vis.plot_model(cnn_restored)

#%%
# plot channels
fct_train = preprocess_input(hind_2000_2019, v, path_data, lead_input)

fct_train = fct_train.sel(latitude = input_lat, longitude = input_lon)

plt.figure()
fct_train.lower_t2m.isel(forecast_time = 0).plot()
plt.figure()
fct_train.upper_t2m.isel(forecast_time = 0).plot()
#%%
# plot targets/labels/terciled obs
plt.figure()
obs_2000_2019_terciled.sel(latitude = output_lat, longitude = output_lon).t2m.isel(forecast_time = 0).plot(col = 'category')
#%%

# plot cdf to illustrate what channels represent

data = obs_2000_2019.t2m.isel(latitude = 40, longitude = 40).values

#sort data
x = np.sort(data)

#calculate CDF values
y = 1. * np.arange(len(data)) / (len(data) - 1)

#plot CDF
plt.figure()
plt.plot(x, y)

#%%

# plot raw forecast
da = xr.open_dataset('../submissions/10/global_prediction_t2m_lead0_smooth_allyears.nc')['t2m']
plt.figure()
da.isel(forecast_time = 0, category = 0).plot(vmin = 0.3, vmax = 0.5, cbar_kwargs={'ticks': [0.3,0.35,0.4,0.45,0.5]})#
plt.title('')
plt.savefig('plots/smooth_prediction.png', dpi = 1200)

# plot smoothed forecast
da = xr.open_dataset('../submissions/10/global_prediction_t2m_lead0_raw_allyears.nc')['t2m']
plt.figure()
da.isel(forecast_time = 0, category = 0).plot(vmin = 0.3, vmax = 0.5, cbar_kwargs={'ticks': [0.3,0.35,0.4,0.45,0.5]})
plt.title('')
plt.savefig('plots/raw_prediction.png', dpi = 1200)
