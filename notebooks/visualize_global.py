# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:41:51 2021

@author: ab6801
"""

#%%



import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Dot, Add, Activation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import xarray as xr
xr.set_options(display_style='text')

import xskillscore as xs



from scripts import skill_by_year, add_year_week_coords
from helper_ml_data import load_data, get_basis, rm_annualcycle, rm_tercile_edges, rm_tercile_edges1, DataGenerator1, single_prediction, skill_by_year_single


import warnings
warnings.simplefilter("ignore")
#%%
path_data = 'local_n'

v= 't2m'
lead_output = 0
#%%

#cnn_restored = keras.models.load_model("CNN_patch_2")


da = xr.open_dataset('../submissions/global_prediction_t2m_lead34_2020_smooth.nc')[v]

obs_2000_2019_terciled = load_data(data = 'obs_terciled_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)[v]

obs_2020_terciled = load_data(data = 'obs_terciled_2020', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)[v]


####visualize RPSS regionally, seasonally...
fct_p = da
obs_p = obs_2020_terciled#.sel(forecast_time=slice('2017','2018'))[v]#terciled_obs

#def compute_rpss():
# climatology
clim_p = xr.DataArray([1/3, 1/3, 1/3], dims='category', coords={'category':['below normal', 'near normal', 'above normal']}).to_dataset(name='tp')
clim_p['t2m'] = clim_p['tp']

clim_p = clim_p[v]

## RPSS
# rps_ML
rps_ML = xs.rps(obs_p, fct_p, category_edges=None, dim=[], input_distributions='p').compute()
# rps_clim
rps_clim = xs.rps(obs_p, clim_p, category_edges=None, dim=[], input_distributions='p').compute()

# rpss
rpss = 1 - (rps_ML / rps_clim)
#%%

plt.figure()
rpss.mean('forecast_time').plot()

plt.figure()
rpss.mean(('latitude','longitude')).plot()

plt.figure()
stacked = rpss.stack( z = ('latitude','longitude')).reset_index("z")
stacked.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

plt.figure()
stacked_nh = rpss.sel(latitude = slice (90,30)).stack( z = ('latitude','longitude')).reset_index("z")
stacked_nh.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

plt.figure()
stacked_t = rpss.sel(latitude = slice (30,-30)).stack( z = ('latitude','longitude')).reset_index("z")
stacked_t.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

plt.figure()
stacked_sh = rpss.sel(latitude = slice (-30,-60)).stack( z = ('latitude','longitude')).reset_index("z")
stacked_sh.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

output_lat = slice(63,52)#8#slice(69,46)#16
output_lon = slice(67,78)

plt.figure()
rpss.sel(latitude = output_lat, longitude = output_lon).mean(('latitude','longitude')).plot()#2018 good skill from jan to mai. 2017 no skill during that period

#%%
# =============================================================================
# 
# trained_patch = fct_train.sel(latitude = input_lat, longitude = input_lon)#.sel(forecast_time = slice('2017','2018'))
# preds_trained_patch = cnn.predict([trained_patch.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
#                                        np.repeat(basis[np.newaxis,:,:],len(trained_patch.forecast_time),axis=0),
#                                        np.repeat(clim_probs[np.newaxis,:,:],len(trained_patch.forecast_time),axis=0)]
#                     ).squeeze()
#     
# preds_trained_patch = Reshape((len(lons),len(lats),3))(preds_trained_patch)
# 
# 
# da_trained_patch = xr.DataArray(
#                 tf.transpose(preds_trained_patch, [0,3,1,2]),
#                 dims=['forecast_time', 'category','longitude', 'latitude'],
#                 coords={'forecast_time': fct_train.forecast_time, 'category' : verif_train.category, 
#                         'latitude': trained_patch.sel(latitude = output_lat, longitude = output_lon).latitude,#slice(8,24)
#                         'longitude': trained_patch.sel(latitude = output_lat, longitude = output_lon).longitude
#                         }
#             )
# da_trained_patch = da_trained_patch.transpose('category','forecast_time','latitude',...)
# da_trained_patch = da_trained_patch.assign_coords(lead_time=lead_output)
# 
# da_trained_patch.isel(forecast_time = 0).plot(col = 'category')
# 
# fct_p = da_trained_patch#g#prediction
# obs_p = obs_2000_2019_terciled.sel(latitude = output_lat, longitude = output_lon)[v]#.sel(forecast_time=slice('2017','2018'))[v]#terciled_obs
# 
# 
# # climatology
# clim_p = xr.DataArray([1/3, 1/3, 1/3], dims='category', coords={'category':['below normal', 'near normal', 'above normal']}).to_dataset(name='tp')
# clim_p['t2m'] = clim_p['tp']
# 
# clim_p = clim_p[v]
# 
# ## RPSS
# # rps_ML
# rps_ML = xs.rps(obs_p, fct_p, category_edges=None, dim=[], input_distributions='p').compute()
# # rps_clim
# rps_clim = xs.rps(obs_p, clim_p, category_edges=None, dim=[], input_distributions='p').compute()
# 
# # rpss
# rpss = 1 - (rps_ML / rps_clim)
# 
# rpss.sel(latitude = output_lat, longitude = output_lon).mean(('latitude','longitude')).plot()
# plt.figure()
# rpss.mean(('latitude','longitude')).rolling(forecast_time = 40).mean().plot(figsize = (10,4))
# plt.figure()
# rpss.groupby('forecast_time.year').mean().mean(('latitude','longitude')).plot()
# plt.figure()
# rpss.groupby('forecast_time.month').mean().mean(('latitude','longitude')).plot()
# =============================================================================

#%%
###########gaussian filtering##################################################
# =============================================================================
# skill_list = []
# 
# for sigma in range(0,10):
#     print(prediction.shape)
#     
#     img = sp.ndimage.gaussian_filter(prediction, sigma=(0,sigma,sigma,0), order=0)
#     
#     da_img = xr.DataArray(
#                     tf.transpose(img, [0,3,1,2]),
#                     dims=['forecast_time', 'category','longitude', 'latitude'],
#                     coords={'forecast_time': fct_predict.forecast_time, 'category' : verif_train.category, 
#                             'latitude': fct_train.latitude,#slice(8,24)
#                             'longitude': fct_train.longitude
#                             }
#                 )
#     da_img = da_img.transpose('category','forecast_time','latitude',...)
#     da_img = da_img.assign_coords(lead_time=lead_output)
#     
#     #da_img.isel(forecast_time = 0).plot(col = 'category')
#     
#     skill = skill_by_year_single(da_img,#.sel(latitude = slice(90,-60)), #does not change the rpss
#                                  obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018'))[v],
#                                  v).RPSS     
#     skill_list.append(skill.values)
#     print(skill)
# 
# skill_df = pd.DataFrame(skill_list, columns = ['2017','2018'])
# skill_df.to_csv('global_skill.csv', sep =';', index = False)  
# =============================================================================

# =============================================================================
# #ufunc could be useful if already converted to xarray and you want to keep the coords
# img = xr.apply_ufunc(sp.ndimage.gaussian_filter, prediction,sigma, input_core_dims=[['forecast_time', 'category'], []],#'forecast_time'', 'category'#'latitude','longitude'
#                      dask = 'allowed', vectorize = True, output_core_dims = [['forecast_time', 'category']])
# #img = sp.ndimage.gaussian_filter(prediction,sigma=sigma,truncate=truncate)# gaussian_kernel(input_gauss, sigma = 1, order = 0)
# =============================================================================
   