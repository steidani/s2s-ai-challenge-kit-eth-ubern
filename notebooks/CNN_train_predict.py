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


import xarray as xr
xr.set_options(display_style='text')

import xskillscore as xs



from scripts import skill_by_year, add_year_week_coords

from helper_ml_data import load_data, get_basis, rm_annualcycle, rm_tercile_edges, rm_tercile_edges1, DataGenerator1
from helper_ml_data import add_valid_time_single, single_prediction, skill_by_year_single


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
ep  = 2


#train domain:
input_lat = slice(81,34)#32
input_lon = slice(49,97)#32

output_lat = slice(63,52)#8          #slice(69,46)#16
output_lon = slice(67,78)#8          #slice(61,85) #16
    
#%%
#read data

path_data = 'local_n'#'server'#
#['../../../../Data/s2s_ai/data', '../../../../Data/s2s_ai']

cases = ['w34', 'w2', 'w234']
j = 'w34'

if j == 'w34':
    # ## Hindcast
    hind_2000_2019_raw = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_output)#['tp','t2m','sm20','sst'])#,'lsm', 'msl', var_list = ['tp','t2m'])#path_data)
    if 'sm20' in hind_2000_2019_raw.keys():
        hind_2000_2019_raw = hind_2000_2019_raw.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
    elif 'msl' in hind_2000_2019_raw.keys():
        hind_2000_2019_raw = hind_2000_2019_raw.isel(meanSea = 0).reset_coords('meanSea', drop = True)
    hind_2000_2019_raw = hind_2000_2019_raw[['t2m']]

elif j =='w2':
            # ## Hindcast
    hind_2000_2019_raw = load_data(data = 'hind_2000-2019', aggregation = 'weekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)#['tp','t2m','sm20','sst'])#,'lsm', 'msl', var_list = ['tp','t2m'])#path_data)
    if 'sm20' in hind_2000_2019_raw.keys():
        hind_2000_2019_raw = hind_2000_2019_raw.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
    elif 'msl' in hind_2000_2019_raw.keys():
        hind_2000_2019_raw = hind_2000_2019_raw.isel(meanSea = 0).reset_coords('meanSea', drop = True)
    hind_2000_2019_raw = [hind_2000_2019_raw[n].rename(f'{n}_2') for n in list(hind_2000_2019_raw.keys())]
    hind_2000_2019_raw = xr.merge(hind_2000_2019_raw)
    hind_2000_2019_raw = hind_2000_2019_raw[['t2m_2']]

elif j == 'w234':  
    hind_2000_2019_raw_34 = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_output)#['tp','t2m','sm20','sst'])#,'lsm', 'msl', var_list = ['tp','t2m'])#path_data)
    if 'sm20' in hind_2000_2019_raw_34.keys():
        hind_2000_2019_raw_34 = hind_2000_2019_raw_34.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
    elif 'msl' in hind_2000_2019_raw_34.keys():
        hind_2000_2019_raw_34 = hind_2000_2019_raw_34.isel(meanSea = 0).reset_coords('meanSea', drop = True)


    hind_2000_2019_raw_2 = load_data(data = 'hind_2000-2019', aggregation = 'weekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)#['tp','t2m','sm20','sst'])#,'lsm', 'msl', var_list = ['tp','t2m'])#path_data)
    if 'sm20' in hind_2000_2019_raw_2.keys():
        hind_2000_2019_raw_2 = hind_2000_2019_raw_2.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
    elif 'msl' in hind_2000_2019_raw_2.keys():
        hind_2000_2019_raw_2 = hind_2000_2019_raw_2.isel(meanSea = 0).reset_coords('meanSea', drop = True)
    
    #new_names = [f'{n}_2' for n in hind_2000_2019_raw_2.keys()]
    #hind_2000_2019_raw_2 = hind_2000_2019_raw_2.rename({hind_2000_2019_raw_2.keys() : new_names})#(hind_2000_2019_raw_2.keys() : new_names)
    
    hind_2000_2019_raw_2 = [hind_2000_2019_raw_2[n].rename(f'{n}_2') for n in list(hind_2000_2019_raw_2.keys())]#.to_dataset()#.rename(f'{n}_2')
    hind_2000_2019_raw_2 = xr.merge(hind_2000_2019_raw_2)#, overwrite = True)
    
    hind_2000_2019_raw = xr.merge([hind_2000_2019_raw_2,hind_2000_2019_raw_34], compat = 'override')
    hind_2000_2019_raw = hind_2000_2019_raw[['t2m','t2m_2']]
# ## Observations corresponding to hindcasts
obs_2000_2019_raw = load_data(data = 'obs_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)
# terciled
obs_2000_2019_terciled_raw = load_data(data = 'obs_terciled_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)

#hind_2000_2019_raw = hind_2000_2019_raw[['t2m','t2m_2']]
hind_2000_2019_raw.assign_coords(lead_time = obs_2000_2019_raw.lead_time)##overwrite lead_time... not ideal but ow strange issues..
    
#%%

#train data:
hind_2000_2019 = hind_2000_2019_raw#.sel(longitude = input_lon, latitude = input_lat)
obs_2000_2019 = obs_2000_2019_raw#.sel(longitude = output_lon, latitude = output_lat)
obs_2000_2019_terciled = obs_2000_2019_terciled_raw#.sel(longitude = output_lon, latitude = output_lat)

# ## masking: only used for label data
#mask: same missing values at all forecast_times
mask = xr.where(obs_2000_2019.notnull(),1,np.nan).mean('forecast_time', skipna = False)


#%%
# ### create datasets

#train
fct_train = hind_2000_2019
  
verif_train = obs_2000_2019_terciled[v]
verif_train = verif_train.where(mask[v].notnull())
  
#%%

#use ensemble mean as input
fct_train = fct_train.mean('realization')

#%%
# ### Remove annual cycle
        
### dont remove annual cycle from land-sea mask!
#if rmcycle == True:
var_list = [i for i in fct_train.keys() if i not in ['lsm',v,f'{v}_2']]

if len(var_list) == 1:
    fct_train[var_list] = rm_annualcycle(fct_train[var_list], fct_train[var_list])[var_list[0]]
elif len(var_list) >1:
    fct_train[var_list] = rm_annualcycle(fct_train[var_list], fct_train[var_list])

#%%
### get tercile features

#if rmterciles == True: 
vars_ = [i for i in fct_train.keys() if i in [v,f'{v}_2']]

#train

#deb = rm_tercile_edges1(fct_train[vars_], fct_train[vars_])##############################maybe use first version of tercile edges that uses precomputed data.

tercile_edges = load_data(data = 'obs_tercile_edges_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_input)
deb = rm_tercile_edges(fct_train[vars_], tercile_edges[vars_])#####buggy but faster

debs = []
for var in list(deb.keys()):
    #print(var)
    debs.append(deb[var].assign_coords( category_edge = ['lower_{}'.format(var),'upper_{}'.format(var)]).to_dataset(dim ='category_edge'))
deb_train = xr.merge(debs)#.map(rm_coords_to_var)


fct_train = xr.merge([fct_train, deb_train])
fct_train = fct_train.drop_vars(vars_)

#%%
####standardization

#if standardization == True:
#normalisation constant for each field
fct_train_std_spatial = fct_train.std(('latitude','longitude')).mean(('forecast_time'))#mean over 'realization', already computed : this is the spatial standard deviation of the ensemble mean

#do the normalization
fct_train = fct_train / fct_train_std_spatial
            
#%%

#up to here is preprocessing of global data


#%%
#train data:
fct_train_patch = fct_train.sel(longitude = input_lon, latitude = input_lat)
verif_train_patch = verif_train.sel(longitude = output_lon, latitude = output_lat)
#%%

# #### compute basis do this only once!, try to reuse it for other the other patches!!!!!
#first_var = list(fct_train.keys())[0]#fct_train[first_var]
basis, lats, lons, n_xy, n_basis = get_basis(verif_train_patch.isel(category = 0), basis_rad) #30#14obs_terciled
##the smaller you choose the radius of the basis functions, the more memory needs to be allocated (a lot more!)

clim_probs = np.log(1/3)*np.ones((n_xy,n_bins))


#%%

# ### create batches
print('create batches')
bs=32

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

#%%

#cnn.save('CNN_patch_2')#creates a new folder with model details
#cnn_restored = keras.models.load_model("CNN_patch_2")



#%%
################################################################################
#prediction for train patch using train

pred_patch = fct_train_patch.sel(forecast_time = slice('2017','2018'))#.sel(lead_time = lead_input)
preds_single = single_prediction(cnn, [pred_patch.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                       np.repeat(basis[np.newaxis,:,:],len(pred_patch.forecast_time),axis=0),
                                       np.repeat(clim_probs[np.newaxis,:,:],len(pred_patch.forecast_time),axis=0)],
                                       lead_input, lons, lats, pred_patch, verif_train_patch, lead_output)    

obs = obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018')).sel(longitude = output_lon, latitude = output_lat)


skill = skill_by_year_single(preds_single, 
                         obs[v],#.sel(lead_time = lead_output),
                         v).RPSS

#%%
##############################################################################
#use global field and select input output patches not equal to train patch 

#xr.map_blocks
fct_predict_patch = fct_train.isel(longitude = slice(0,32), latitude = slice(20,52)).sel(forecast_time = slice('2017','2018'))
obs_predict = obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018')).isel(longitude = slice(0,32), latitude = slice(20,52)).isel(latitude = slice(12,20),longitude = slice(12,20))#.sel(longitude = output_lon, latitude = output_lat)
#%%

preds = cnn.predict([fct_predict_patch.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                       np.repeat(basis[np.newaxis,:,:],len(pred_patch.forecast_time),axis=0),
                                       np.repeat(clim_probs[np.newaxis,:,:],len(fct_predict_patch.forecast_time),axis=0)]
                    ).squeeze()
    
preds = Reshape((len(lons),len(lats),3))(preds)
preds = tf.transpose(preds, [0,3,1,2])
    
#%%
da = xr.DataArray(
                preds,
                dims=['forecast_time', 'category','longitude', 'latitude'],
                coords={'forecast_time': fct_predict_patch.forecast_time, 'category' : verif_train_patch.category, 
                        'latitude': fct_predict_patch.isel(latitude = slice(12,20)).latitude,#slice(8,24)
                        'longitude': fct_predict_patch.isel(longitude = slice(12,20)).longitude
                        }
            )
da = da.transpose('category','forecast_time','latitude',...)
da = da.assign_coords(lead_time=lead_output)

preds_masked = da.where(mask[v].notnull())

da = add_valid_time_single(da, obs.lead_time.values)

#%%

#"transfer learning" does not work.
skill = skill_by_year_single(da, 
                         obs_predict[v],#.sel(lead_time = lead_output),
                         v).RPSS
#%%
##############################################################################
#create prediction for the whole globe using the model trained only on a single patch

#######################domain
# =============================================================================
# print(np.arange(90,-90,-16))
# 
# hind_2000_2019.isel(latitude = slice(0,121,16)).latitude.values
# Out[258]: array([ 90.,  66.,  42.,  18.,  -6., -30., -54.#last no needed, because dont have to predict at south pole, -78.])
#                  
# hind_2000_2019.isel(longitude = slice(0,240,16)).longitude.values
# Out[260]: 
# array([  0.,  24.,  48.,  72.,  96., 120., 144., 168., 192., 216., 240.,
#        264., 288., 312., 336.])
# =============================================================================

#http://xarray.pydata.org/en/stable/generated/xarray.DataArray.pad.html

#%%
#####pad the earth by the required amount
#only works for square input and output
#1.5 is the resolution of the data

pad = int((input_dims - output_dims)/2)
print(pad)

###create padding for north pole
hind_shift = fct_train.pad(pad_width = {'longitude' : (0,120)}, mode = 'wrap').shift({'longitude' : 120}).isel(longitude = slice(120, 120 + int(360/1.5)))
hind_shift_pad = hind_shift.pad(pad_width = {'latitude' : (pad,0)}, mode = 'reflect')
shift_pad = hind_shift_pad.isel(latitude = slice (0,pad))
#plt.figure()
#plt.imshow(shift_pad.isel(forecast_time = 0).lower_t2m.values)

###add north pole padding to ftc_train
hind_lat_pad = xr.concat([shift_pad,  fct_train], dim = 'latitude')
#plt.figure()
#plt.imshow(hind_lat_pad.isel(forecast_time = 0).lower_t2m.values)

### pad in the east-west direction
hind_padded = hind_lat_pad.pad(pad_width = {'longitude' : (pad,pad)}, mode = 'wrap')
#plt.figure()
#plt.imshow(hind_padded.isel(forecast_time = 0).lower_t2m.values)

#%%

#select years to predict for
fct_predict = hind_padded.sel(forecast_time = slice('2017','2018'))

#initialize 1/3 matrix to save predictions
prediction = np.ones(shape = (len(fct_predict.forecast_time), int(360/1.5), int(180/1.5) + 1,3))*1/3
print(prediction.shape)

#iterate over global and create predictions patchwise
for lat_i in range(0,int(150/1.5),output_dims):
    print(lat_i)
    #lat_i = 0
    patch_lat = fct_predict.isel(latitude = slice(lat_i, lat_i + input_dims))
    for lon_i in range(0,int(360/1.5), output_dims):
        print(lon_i)
        #lon_i = 0
        patch = patch_lat.isel(longitude = slice(lon_i, lon_i + input_dims))
        
        print(patch.sizes)
        
        preds = cnn.predict([patch.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                       np.repeat(basis[np.newaxis,:,:],len(patch.forecast_time),axis=0),
                                       np.repeat(clim_probs[np.newaxis,:,:],len(patch.forecast_time),axis=0)]
                    ).squeeze()
    
        preds = Reshape((len(lons),len(lats),3))(preds)
    
        #preds = tf.transpose(preds, [0,3,1,2])
        
        prediction[:,lon_i:(lon_i + output_dims), lat_i:(lat_i + output_dims), :] = preds
        
# =============================================================================
# #plot predictions     
# plt.figure()
# plt.imshow(prediction[0,:,:,0].transpose())
# plt.colorbar()
# =============================================================================

#%%
# add coordinates and save predictions
da = xr.DataArray(
                tf.transpose(prediction, [0,3,1,2]),
                dims=['forecast_time', 'category','longitude', 'latitude'],
                coords={'forecast_time': fct_predict.forecast_time, 'category' : verif_train.category, 
                        'latitude': fct_train.latitude,#slice(8,24)
                        'longitude': fct_train.longitude
                        }
            )
da = da.transpose('category','forecast_time','latitude',...)
da = da.assign_coords(lead_time=lead_output)

#better to save as dataframe and not as dataarray
da.to_netcdf('../submissions/global_prediction_t2m_lead34_1718.nc')

#%%
#
#da = xr.open_dataset('../submissions/global_prediction_t2m_lead34_1718_20epochs.nc').to_array(dim = 'var').isel(var = 0).reset_coords('var', drop= True)

#%%


da.isel(forecast_time = 0).plot(col = 'category')
skill = skill_by_year_single(da,#.sel(latitude = slice(90,-60)), #does not change the rpss
                             obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018'))[v],
                             v).RPSS     
print(skill)

#%%

# patchwise predictions result in edgy forecasts, use gaussian filtering to smooth the predictions
# one additional hyperparameter to optimize: sigma
#local skill
import scipy as sp
import scipy.ndimage

#sp.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)
sigma = 9# best sigma is 9 for the years 2017/2018

skill_list = []

for sigma in range(0,10):
# =============================================================================
# img = xr.apply_ufunc(sp.ndimage.gaussian_filter, prediction,sigma, input_core_dims=[['forecast_time', 'category'], []],#'forecast_time'', 'category'#'latitude','longitude'
#                      dask = 'allowed', vectorize = True, output_core_dims = [['forecast_time', 'category']])
# #img = sp.ndimage.gaussian_filter(prediction,sigma=sigma,truncate=truncate)# gaussian_kernel(input_gauss, sigma = 1, order = 0)
# =============================================================================
    print(prediction.shape)
    
    img = sp.ndimage.gaussian_filter(prediction, sigma=(0,sigma,sigma,0,), order=0)
    
    da_img = xr.DataArray(
                    tf.transpose(img, [0,3,1,2]),
                    dims=['forecast_time', 'category','longitude', 'latitude'],
                    coords={'forecast_time': fct_predict.forecast_time, 'category' : verif_train.category, 
                            'latitude': fct_train.latitude,#slice(8,24)
                            'longitude': fct_train.longitude
                            }
                )
    da_img = da_img.transpose('category','forecast_time','latitude',...)
    da_img = da_img.assign_coords(lead_time=lead_output)
    
    #da_img.isel(forecast_time = 0).plot(col = 'category')
    
    skill = skill_by_year_single(da_img,#.sel(latitude = slice(90,-60)), #does not change the rpss
                                 obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018'))[v],
                                 v).RPSS     
    skill_list.append(skill.values)
    print(skill)

skill_df = pd.DataFrame(skill_list, columns = ['2017','2018'])
skill_df.to_csv('global_skill.csv', sep =';', index = False)   
#%%

####visualize RPSS regionally, seasonally...
fct_p = da_img#prediction
obs_p = obs_2000_2019_terciled.sel(forecast_time=slice('2017','2018'))[v]#terciled_obs


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

rpss.mean('forecast_time').plot()
rpss.mean(('latitude','longitude')).plot()
stacked = rpss.stack( z = ('latitude','longitude')).reset_index("z")
stacked.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

stacked_nh = rpss.sel(latitude = slice (90,30)).stack( z = ('latitude','longitude')).reset_index("z")
stacked_nh.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);


stacked_t = rpss.sel(latitude = slice (30,-30)).stack( z = ('latitude','longitude')).reset_index("z")
stacked_t.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

stacked_sh = rpss.sel(latitude = slice (-30,-60)).stack( z = ('latitude','longitude')).reset_index("z")
stacked_sh.plot.line(x= 'forecast_time', hue = 'z', add_legend = False);

output_lat = slice(63,52)#8#slice(69,46)#16
output_lon = slice(67,78)

rpss.sel(latitude = output_lat, longitude = output_lon).mean(('latitude','longitude')).plot()#2018 good skill from jan to mai. 2017 no skill during that period

#%%

trained_patch = fct_train.sel(latitude = input_lat, longitude = input_lon)#.sel(forecast_time = slice('2017','2018'))
preds_trained_patch = cnn.predict([trained_patch.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                       np.repeat(basis[np.newaxis,:,:],len(trained_patch.forecast_time),axis=0),
                                       np.repeat(clim_probs[np.newaxis,:,:],len(trained_patch.forecast_time),axis=0)]
                    ).squeeze()
    
preds_trained_patch = Reshape((len(lons),len(lats),3))(preds_trained_patch)


da_trained_patch = xr.DataArray(
                tf.transpose(preds_trained_patch, [0,3,1,2]),
                dims=['forecast_time', 'category','longitude', 'latitude'],
                coords={'forecast_time': fct_train.forecast_time, 'category' : verif_train.category, 
                        'latitude': trained_patch.sel(latitude = output_lat, longitude = output_lon).latitude,#slice(8,24)
                        'longitude': trained_patch.sel(latitude = output_lat, longitude = output_lon).longitude
                        }
            )
da_trained_patch = da_trained_patch.transpose('category','forecast_time','latitude',...)
da_trained_patch = da_trained_patch.assign_coords(lead_time=lead_output)

da_trained_patch.isel(forecast_time = 0).plot(col = 'category')

fct_p = da_trained_patch#g#prediction
obs_p = obs_2000_2019_terciled.sel(latitude = output_lat, longitude = output_lon)[v]#.sel(forecast_time=slice('2017','2018'))[v]#terciled_obs


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

rpss.sel(latitude = output_lat, longitude = output_lon).mean(('latitude','longitude')).plot()
plt.figure()
rpss.mean(('latitude','longitude')).rolling(forecast_time = 40).mean().plot(figsize = (10,4))
plt.figure()
rpss.groupby('forecast_time.year').mean().mean(('latitude','longitude')).plot()
plt.figure()
rpss.groupby('forecast_time.month').mean().mean(('latitude','longitude')).plot()

#%%

##read 2020 data and create prediction for 2020 data.
        
