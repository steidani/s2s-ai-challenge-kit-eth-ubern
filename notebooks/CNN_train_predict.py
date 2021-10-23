# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:30:06 2021

train and predict
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
# ### create datasets
fct_train = hind_2000_2019
  
verif_train = obs_2000_2019_terciled[v]
verif_train = verif_train.where(mask[v].notnull())

# preprocess input: compute and standardize features
fct_train = preprocess_input(fct_train, v, path_data, lead_input)

#up to here is preprocessing of global data

#vars to get back dataset coords after prediction
global_lats = fct_train.latitude
global_lons = fct_train.longitude
lead_output_coords = fct_train.lead_time


#%%
#train data:
fct_train_patch = fct_train.sel(longitude = input_lon, latitude = input_lat)
verif_train_patch = verif_train.sel(longitude = output_lon, latitude = output_lat)
#%%

# #### compute basis
basis, lats, lons, n_xy, n_basis = get_basis(verif_train_patch.isel(category = 0), basis_rad)
##the smaller you choose the radius of the basis functions, the more memory needs to be allocated (a lot more!)

clim_probs = np.log(1/3)*np.ones((n_xy,n_bins))

#%%


def train_patch(fct_train_patch, verif_train_patch, basis, clim_probs, bs):

    # ### create batches
    print('create batches')
    dg_train = DataGenerator1(fct_train_patch, verif_train_patch, 
                             basis = basis, clim_probs = clim_probs,
                             batch_size=bs, load=True)
    
    print('finished creating batches')
    
    #%%
    
    # CNN: slightly adapted from Scheuerer et al 2020.
            
    inp_imgs = Input(shape=(dg_train[0][0][0].shape[1], dg_train[0][0][0].shape[2],dg_train[0][0][0].shape[3],)) #fcts4121,240,
    inp_basis = Input(shape=(n_xy,n_basis)) #basis
    inp_cl = Input(shape=(n_xy,n_bins,)) #climatology
    
    c = Conv2D(4, (3,3), activation='elu')(inp_imgs)
    c = MaxPooling2D((2,2))(c)
    #c = Conv2D(8, (3,3), activation='elu')(c)
    #c = MaxPooling2D((2,2))(c)
    x = Flatten()(c)
    for h in hidden_nodes: 
        x = Dropout(dropout_rate)(x)
        x = Dense(h, activation='elu')(x)
    x = Dense(n_bins*n_basis, activation='elu')(x)
    x = Reshape((n_bins,n_basis))(x)
    z = Dot(axes=2)([inp_basis, x])     # Tensor product with basis functions
    #z = Add()([z, inp_cl])              # Add (log) probability anomalies to log climatological probabilities 
    out = Activation('softmax')(z)
    
    
    cnn = Model(inputs=[inp_imgs, inp_basis, inp_cl], outputs=out)
    #cnn.summary()
    
    
    cnn.compile(loss= keras.losses.CategoricalCrossentropy(label_smoothing = smoothing), metrics=['accuracy'], optimizer=keras.optimizers.Adam(1e-4))#'adam')#my_loss_rps
    #%%
    
    hist =cnn.fit(dg_train, epochs=ep, shuffle = False)   

    return cnn
#%%

cnn = train_patch(fct_train_patch, verif_train_patch, basis, clim_probs, bs)

cnn.save(f'CNN_patch_34_{v}_{ep}')#creates a new folder with model details
#cnn_restored = keras.models.load_model("CNN_patch_2")
#%%

##############################################################################
#predict for training data
#####pad the earth by the required amount
hind_padded = pad_earth(fct_train, input_dims, output_dims)

#select years to predict for
fct_predict = hind_padded#.sel(forecast_time = slice('2017','2018'))
#%%

### predict globally using the trained local model    
prediction = slide_predict(fct_predict, input_dims, output_dims, cnn, basis, clim_probs)

da = add_coords(prediction, fct_predict, global_lats, global_lons, lead_output_coords)
da.to_dataset(name = v).to_netcdf('../submissions/global_prediction_t2m_lead34_raw_allyears.nc')

#plot predictions     
plt.figure()
plt.imshow(prediction[0,:,:,0].transpose())
plt.colorbar()

#%%
#smoothed predictions
# patchwise predictions result in edgy forecasts, use gaussian filtering to smooth the predictions
# one additional hyperparameter to optimize: sigma
prediction_smoothed = sp.ndimage.gaussian_filter(prediction, sigma=(0,sigma,sigma,0), order=0)

da = add_coords(prediction_smoothed, fct_predict, global_lats, global_lons, lead_output_coords)
da.to_dataset(name = v).to_netcdf('../submissions/global_prediction_t2m_lead34_smoothed_allyears.nc')

#%%
#
#da = xr.open_dataset('../submissions/global_prediction_t2m_lead34_1718_new2.nc')[v]
#da = xr.open_dataset('../submissions/global_prediction_t2m_lead34_1718_20epochs.nc').to_array(dim = 'var').isel(var = 0).reset_coords('var', drop= True)

# =============================================================================
# da.isel(forecast_time = 0).plot(col = 'category')
# skill = skill_by_year_single(da,#.sel(latitude = slice(90,-60)), #does not change the rpss
#                              obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018'))[v],
#                              v).RPSS     
# print(skill)
# =============================================================================
 
#%%%
###############################################################################
##read 2020 data and create prediction for 2020 data.
        
fct_2020 = load_data(data = 'forecast_2020', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)
fct_2020 = clean_coords(fct_2020)[['t2m']]

# preprocess input: compute and standardize features
fct_2020 = preprocess_input(fct_2020, v, path_data, lead_input)

fct_2020_padded = pad_earth(fct_2020, input_dims, output_dims)

pred_2020 = slide_predict(fct_2020_padded, input_dims, output_dims, cnn, basis, clim_probs)

#save raw predictions
da = add_coords(pred_2020, fct_2020, global_lats, global_lons, lead_output_coords)
da.to_dataset(name = v).to_netcdf('../submissions/global_prediction_t2m_lead34_2020_raw.nc')

pred_2020_smoothed = sp.ndimage.gaussian_filter(pred_2020, sigma=(0,sigma,sigma,0), order=0)
#convert to dataset and save
da_smooth = add_coords(pred_2020_smoothed, fct_2020, global_lats, global_lons, lead_output_coords)
da_smooth.to_dataset(name = v).to_netcdf('../submissions/global_prediction_t2m_lead34_2020_smooth.nc')



