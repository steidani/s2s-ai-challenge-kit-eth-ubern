# -*- coding: utf-8 -*-
"""
Functions to load and prepare data for training/validating/testing Ml models
"""
import numpy as np
import xarray as xr
from scripts import add_year_week_coords
xr.set_options(display_style='text')

import xskillscore as xs

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Dot, Add, Activation


### set seed to make shuffling in datagenerator reproducible
np.random.seed(42)

import os
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

import random as rn
#set all random seeds
os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(1)
rn.seed(1254)
tf.random.set_seed(89)

def load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = 'server', var_list = ['tp','t2m','sm20','sst']):
    """
    Parameters
    ----------
    data: string, {'hind_2000-2019', 'obs_2000-2019', 'obs_terciled_2000-2019','obs_tercile_edges_2000-2019', 'forecast_2020', 'obs_2020', 'obs_terciled_2020'}
    aggregation: string, {'biweekly','weekly'}
    path: string or 2-element list, {'server', 'local_n', user-defined path to the data}, 
                    server assumes that this function is called within a file in the home directory on the compute server
    var_list: list, list of variables
    """
    
    if path == 'local_n':
        cache_path = '../template/data' 
        path_add_vars = '../data'
    elif path == 'server':
        cache_path = '../../../../Data/s2s_ai/data'
        path_add_vars = '../../../../Data/s2s_ai'
    else: 
        if len(path) !=2:
            return 'please specify a valid path'
        cache_path = path[0]
        path_add_vars = path[1]
    
    if data == 'hind_2000-2019':
        dat_list = []
        for var in var_list:
            if (var == 'tp') or (var == 't2m'):
                if aggregation == 'biweekly':
                    dat_item = xr.open_zarr('{}/ecmwf_hindcast-input_2000-2019_{}_deterministic.zarr'.format(cache_path, aggregation), consolidated=True)
                else:
                    dat_item = xr.open_zarr('{}/ecmwf_hindcast-input_2000-2019_{}_deterministic.zarr'.format(path_add_vars, aggregation), consolidated=True)
                var_list = [i for i in var_list if i not in ['tp', 't2m']]
            else:
                dat_item = xr.open_zarr('{}/ecmwf_hindcast-input_2000-2019_{}_deterministic_{}.zarr'.format(path_add_vars, aggregation,var), consolidated=True)
            dat_list.append(dat_item)
        dat = xr.merge(dat_list)
    
    elif data == 'obs_2000-2019':
        dat = xr.open_zarr(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_deterministic.zarr', consolidated=True)  
    
    elif data == 'obs_terciled_2000-2019':
        dat = xr.open_zarr(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled.zarr', consolidated=True)
        
    elif data == 'obs_tercile_edges_2000-2019':
        dat = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_tercile-edges.nc')
        
    elif data == 'forecast_2020':
        dat_list = []
        for var in var_list:
            if (var == 'tp') or (var == 't2m'):
                if aggregation == 'biweekly':
                    dat_item = xr.open_zarr('{}/ecmwf_forecast-input_2020_{}_deterministic.zarr'.format(cache_path, aggregation), consolidated=True)
                else:
                    dat_item = xr.open_zarr('{}/ecmwf_forecast-input_2020_{}_deterministic.zarr'.format(path_add_vars, aggregation), consolidated=True)
                var_list = [i for i in var_list if i not in ['tp', 't2m']]
            else:
                dat_item = xr.open_zarr('{}/ecmwf_forecast-input_2020_{}_deterministic_{}.zarr'.format(path_add_vars, aggregation,var), consolidated=True)
            dat_list.append(dat_item)
        dat = xr.merge(dat_list)
    
    elif data == 'obs_2020':
        dat = xr.open_zarr(f'{cache_path}/forecast-like-observations_2020_biweekly_deterministic.zarr', consolidated=True)  
    
    elif data == 'obs_terciled_2020':
        dat = xr.open_dataset(f'{cache_path}/forecast-like-observations_2020_biweekly_terciled.nc')
    else: print("specified data name is not valid")
    return dat

def clean_coords(dat):
    #remove unnecessary coords
    if 'sm20' in dat.keys():
        dat = dat.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
    elif 'msl' in dat.keys():
        dat = dat.isel(meanSea = 0).reset_coords('meanSea', drop = True)
    return dat
    
def rm_annualcycle(ds, ds_train):
    #remove annual cycle for each location 
    
    ds = add_year_week_coords(ds)
    ds_train = add_year_week_coords(ds_train)
    
    if 'realization' in ds_train.coords:#always use train data to compute the annual cycle
        ens_mean = ds_train.mean('realization')
    else:
        ens_mean = ds_train

    ds_stand = ds - ens_mean.groupby('week').mean(['forecast_time'])

    ds_stand = ds_stand.sel({'week' : ds.coords['week']})
    ds_stand = ds_stand.drop(['week','year'])
    ds_stand
    return ds_stand

def rm_tercile_edges(ds, tercile_edges):
    #remove annual cycle for each location 
    
    ds = add_year_week_coords(ds)

    ds_stand = tercile_edges - ds

    ds_stand = ds_stand.sel({'week' : ds.coords['week']})
    ds_stand = ds_stand.drop(['week','year'])
    ds_stand
    return ds_stand 


def rm_tercile_edges1(ds, ds_train):
    #remove annual cycle for each location 
    ds = add_year_week_coords(ds)
    ds_train = add_year_week_coords(ds_train)
    
    tercile_edges = ds_train.chunk({'forecast_time':-1,'longitude':'auto'}).groupby('week').quantile(q=[1./3.,2./3.], dim='forecast_time').rename({'quantile':'category_edge'}).astype('float32')#groupby('week').mean(['forecast_time'])

    ds_stand = tercile_edges - ds

    ds_stand = ds_stand.sel({'week' : ds.coords['week']})
    ds_stand = ds_stand.drop(['week','year'])
    #ds_stand
    return ds_stand

def preprocess_input(fct,v, path_data, lead_input):
    """preprocess the ensemble forecast input:
        - compute ensemble mean
        - remove annual cycle from non-target features
        - subtract tercile edges from target variable
        - standardize using the spatial global standard deviation
        
        PARAMETERS:
            fct: (xarray DataSet) contains ensemble forecasts for one lead time
            v: (str) target variable
            lead_input: (int) lead time in fct (for isel)
        RETURNS:
            fct: (xarray DataSet) contains standardized features
        """

    #use ensemble mean
    fct = fct.mean('realization')

    ####remove annual cycle from non-target features       
    #dont remove annual cycle from land-sea mask and target variable
    var_list = [i for i in fct.keys() if i not in ['lsm',v,f'{v}_2']]#v_2: for week two data
    if len(var_list) == 1:
        fct[var_list] = rm_annualcycle(fct[var_list], fct[var_list])[var_list[0]]
    elif len(var_list) >1:
        fct[var_list] = rm_annualcycle(fct[var_list], fct[var_list])

    #####subtract tercile edges from target variable
    vars_ = [i for i in fct.keys() if i in [v,f'{v}_2']]
    tercile_edges = load_data(data = 'obs_tercile_edges_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_input)
    deb = rm_tercile_edges(fct[vars_], tercile_edges[vars_])
    #deb = rm_tercile_edges1(fct[vars_], fct[vars_])#computes terciles on the fly, by respecting train val split, not necessary here
   
    #rename the new tercile features
    debs = []
    for var in list(deb.keys()):
        debs.append(deb[var].assign_coords(category_edge = ['lower_{}'.format(var),'upper_{}'.format(var)]).to_dataset(dim ='category_edge'))
    deb_renamed = xr.merge(debs)
    
    #merge all features and drop original target variable feature
    fct = xr.merge([fct, deb_renamed])
    fct = fct.drop_vars(vars_)

    ####standardization
    fct_std_spatial = fct.std(('latitude','longitude')).mean(('forecast_time'))#this is the spatial standard deviation of the ensemble mean
    fct = fct / fct_std_spatial
    
    return fct


def get_basis(out_field, r_basis):
    """returns a set of basis functions for the input field, adapted from Scheuerer et al. 2020.

    PARAMETERS:
    out_field : (xarray DataArray) basis functions for these lat lon coords will be created
    r_basis : (int) radius of support of basis functions, 
                    the distance between centers of basis functions is half this radius,
                    should be choosen depending on input field size.
    
    RETURNS:
    basis : 
    lats : lats of input field
    lons : lons of input field
    n_xy : number of grid points in input field
    n_basis : number of basis functions
    """  
    
    #r_basis = 14 #radius of support of basis functions
    dist_basis = r_basis/2 #distance between centers of basis functions

    lats = out_field.latitude
    lons = out_field.longitude
    
    #number of basis functions
    n_basis = int(np.ceil((lats[0] - lats[-1])/dist_basis + 1)*np.ceil((lons[-1] - lons[0])/dist_basis + 1))

    #grid coords
    lon_np = lons
    lat_np = lats
    
    length_lon = len(lon_np)
    length_lat = len(lat_np)
    
    lon_np = np.outer(lon_np, np.ones(length_lat)).reshape(int(length_lon * length_lat))
    lat_np = np.outer(lat_np, np.ones(length_lon)).reshape(int(length_lon * length_lat))

    #number of grid points
    n_xy = int(length_lon*length_lat)

    #centers of basis functions
    lon_ctr = np.arange(lons[0],lons[-1] + dist_basis,dist_basis)
    length_lon_ctr = len(lon_ctr) #number of center points in lon direction

    lat_ctr = np.arange(lats[0],lats[-1] - dist_basis,- dist_basis)
    length_lat_ctr = len(lat_ctr) #number of center points in lat direction

    lon_ctr = np.outer(lon_ctr, np.ones(length_lat_ctr)).reshape(int(n_basis))
    lat_ctr = np.outer(np.ones(length_lon_ctr), lat_ctr).reshape(int(n_basis))

    #compute distances between fct grid and basis function centers
    dst_lon = np.abs(np.subtract.outer(lon_np,lon_ctr).reshape(len(lons),len(lats),n_basis))#10,14
    dst_lon = np.swapaxes(dst_lon, 0, 1)
    dst_lat = np.abs(np.subtract.outer(lat_np,lat_ctr).reshape(len(lats),len(lons),n_basis))#'14,10'

    dst = np.sqrt(dst_lon**2+dst_lat**2)
    dst = np.swapaxes(dst, 0, 1).reshape(n_xy,n_basis)

    #define basis functions
    basis = np.where(dst>r_basis,0.,(1.-(dst/r_basis)**3)**3)#main step, zero outside, 
    basis = basis/np.sum(basis,axis=1)[:,None]#normalization at each grid point
    #nbs = basis.shape[1]
    
    return basis, lats, lons, n_xy, n_basis

class DataGenerator1(keras.utils.Sequence):
    # this data generator does not select the lead time, it expects to get dataarrays that are 1D/ or dont have a lead_time coordinate
    # build a data generator, ow it takes too long to convert to numpy. its also probably too large for an efficient training.
    # creating the data generators takes too much time, probably because everything has to be opened because of rm_annualcycle
    
    #returns: list with batches (len(dg_train)= no of batches), each batch consists of two elements X and y
    def __init__(self, fct, verif,  basis, clim_probs, batch_size=32, shuffle=True, load=True):#, ##always have to supply verif, also for test data
                 #mean_fct=None, std_fct=None, mean_verif = None, std_verif = None, window_size = 25):
                     #lead_input, lead_output,
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Args:
            fct: forecasts from S2S models: xr.DataArray (xr.Dataset doesnt work properly)
            verif: not true#observations with same dimensionality (xr.Dataset doesnt work properly)
            lead_time: Lead_time as in model
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
            
        Todo:
        - use number in a better way, now uses only ensemble mean forecast
        - dont use .sel(lead_time=lead_time) to train over all lead_time at once
        - be sensitive with forecast_time, pool a few around the weekofyear given
        - use more variables as predictors
        - predict more variables
        """
        
                
        self.batch_size = batch_size
        self.shuffle = shuffle
        #self.lead_input = lead_input
        #self.lead_output = lead_output
        
        self.basis = basis
        self.clim_probs = clim_probs
        
        ###remove annual cycle here
        ### add more variables
        
        self.fct_data = fct.transpose('forecast_time', ...)#.sel(lead_time=lead_input)
        #self.fct_mean = self.fct_data.mean('forecast_time').compute() if mean_fct is None else mean_fct.sel(lead_time = lead)
        #self.fct_std = self.fct_data.std('forecast_time').compute() if std_fct is None else std_fct.sel(lead_time = lead)
        
        self.verif_data = verif.transpose('forecast_time', ...)#.sel(lead_time=lead_output)
        #self.verif_mean = self.verif_data.mean('forecast_time').compute() if mean_verif is None else mean_verif.sel(lead_time = lead)
        #self.verif_std = self.verif_data.std('forecast_time').compute() if std_verif is None else std_verif.sel(lead_time = lead)

        # Normalize
        #self.fct_data = (self.fct_data - self.fct_mean) / self.fct_std
        #self.verif_data = (self.verif_data - self.verif_mean) / self.verif_std
    
        
        self.n_samples = self.fct_data.forecast_time.size
        #self.forecast_time = self.fct_data.forecast_time
        #self.n_lats = self.fct_data.latitude.size - self.window_size
        #self.n_lons = self.fct_data.longitude.size - self.window_size

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load:
            # print('Loading data into RAM')
            self.fct_data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        #lats = self.lats [i * self.batch_size:(i + 1) * self.batch_size]
        #lons = self.lons[i * self.batch_size:(i + 1) * self.batch_size]
        ##data comes in a row, not randomly chosen from within train data, if shuffled beforehand,--> stays shuffled because of isel
        # got all nan if nans not masked
        X_x = self.fct_data.isel(forecast_time=idxs).fillna(0.).to_array().transpose('forecast_time', ...,'variable').values#.values
        
        
        X_basis = np.repeat(self.basis[np.newaxis,:,:],len(idxs),axis=0)
        X_clim =  np.repeat(self.clim_probs[np.newaxis,:,:],len(idxs),axis=0)#self.batch_size
        
        X = [X_x, X_basis, X_clim]
        
        #X = self.fct_data.isel(forecast_time=idxs).isel(latitude = slice(lats,lats + self.window_size), 
         #                                               longitude = slice(lons,lons + self.window_size)).fillna(0.).values
        #x_coords = (math.ceil((lats + self.window_size)/2),math.ceil((lons + self.window_size)/2))
        y = self.verif_data.isel(forecast_time=idxs).stack(Z = ['latitude','longitude']).transpose('forecast_time','Z',...).fillna(0.).values
        #y = self.verif_data.isel(forecast_time=idxs).fillna(0.).values
        #y = self.verif_data.isel(forecast_time=idxs).isel(latitude = x_coords[0], 
         #                                                 longitude = x_coords[1]).fillna(0.).values
        
        return X, y # x_coords,

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        #self.lats = np.arange(self.n_lats)
        #self.lons = np.arange(self.n_lons)
        if self.shuffle == True: ###does this make sense here?
            np.random.shuffle(self.idxs)#in place 
            #np.random.shuffle(self.lats)
            #np.random.shuffle(self.lons)
            
class DataGenerator(keras.utils.Sequence):
    # build a data generator, ow it takes too long to convert to numpy. its also probably too large for an efficient training.
    # creating the data generators takes too much time, probably because everything has to be opened because of rm_annualcycle
    
    #returns: list with batches (len(dg_train)= no of batches), each batch consists of two elements X and y
    def __init__(self, fct, verif, lead_input, lead_output, basis, clim_probs, batch_size=32, shuffle=True, load=True):#, ##always have to supply verif, also for test data
                 #mean_fct=None, std_fct=None, mean_verif = None, std_verif = None, window_size = 25):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Args:
            fct: forecasts from S2S models: xr.DataArray (xr.Dataset doesnt work properly)
            verif: not true#observations with same dimensionality (xr.Dataset doesnt work properly)
            lead_time: Lead_time as in model
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
            
        Todo:
        - use number in a better way, now uses only ensemble mean forecast
        - dont use .sel(lead_time=lead_time) to train over all lead_time at once
        - be sensitive with forecast_time, pool a few around the weekofyear given
        - use more variables as predictors
        - predict more variables
        """
        
                
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_input = lead_input
        self.lead_output = lead_output
        
        self.basis = basis
        self.clim_probs = clim_probs
        
        ###remove annual cycle here
        ### add more variables
        
        self.fct_data = fct.transpose('forecast_time', ...).sel(lead_time=lead_input)
        #self.fct_mean = self.fct_data.mean('forecast_time').compute() if mean_fct is None else mean_fct.sel(lead_time = lead)
        #self.fct_std = self.fct_data.std('forecast_time').compute() if std_fct is None else std_fct.sel(lead_time = lead)
        
        self.verif_data = verif.transpose('forecast_time', ...).sel(lead_time=lead_output)
        #self.verif_mean = self.verif_data.mean('forecast_time').compute() if mean_verif is None else mean_verif.sel(lead_time = lead)
        #self.verif_std = self.verif_data.std('forecast_time').compute() if std_verif is None else std_verif.sel(lead_time = lead)

        # Normalize
        #self.fct_data = (self.fct_data - self.fct_mean) / self.fct_std
        #self.verif_data = (self.verif_data - self.verif_mean) / self.verif_std
    
        
        self.n_samples = self.fct_data.forecast_time.size
        #self.forecast_time = self.fct_data.forecast_time
        #self.n_lats = self.fct_data.latitude.size - self.window_size
        #self.n_lons = self.fct_data.longitude.size - self.window_size

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load:
            # print('Loading data into RAM')
            self.fct_data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        #lats = self.lats [i * self.batch_size:(i + 1) * self.batch_size]
        #lons = self.lons[i * self.batch_size:(i + 1) * self.batch_size]
        ##data comes in a row, not randomly chosen from within train data, if shuffled beforehand,--> stays shuffled because of isel
        # got all nan if nans not masked
        X_x = self.fct_data.isel(forecast_time=idxs).fillna(0.).to_array().transpose('forecast_time', ...,'variable').values#.values
        
        
        X_basis = np.repeat(self.basis[np.newaxis,:,:],len(idxs),axis=0)
        X_clim =  np.repeat(self.clim_probs[np.newaxis,:,:],len(idxs),axis=0)#self.batch_size
        
        X = [X_x, X_basis, X_clim]
        
        #X = self.fct_data.isel(forecast_time=idxs).isel(latitude = slice(lats,lats + self.window_size), 
         #                                               longitude = slice(lons,lons + self.window_size)).fillna(0.).values
        #x_coords = (math.ceil((lats + self.window_size)/2),math.ceil((lons + self.window_size)/2))
        y = self.verif_data.isel(forecast_time=idxs).stack(Z = ['latitude','longitude']).transpose('forecast_time','Z',...).fillna(0.).values
        #y = self.verif_data.isel(forecast_time=idxs).fillna(0.).values
        #y = self.verif_data.isel(forecast_time=idxs).isel(latitude = x_coords[0], 
         #                                                 longitude = x_coords[1]).fillna(0.).values
        
        return X, y # x_coords,

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        #self.lats = np.arange(self.n_lats)
        #self.lons = np.arange(self.n_lons)
        if self.shuffle == True: ###does this make sense here?
            np.random.shuffle(self.idxs)#in place 
            #np.random.shuffle(self.lats)
            #np.random.shuffle(self.lons)
 
            
def pad_earth(fct, input_dims, output_dims):
    """pad global imput field east, north, west

    only works for square input and output
    1.5 is the resolution of the data
    
    Parameters:
    ----------
    fct: (xarray DataSet)
    input_dims: (int)
    output_dims: (int)
    """
    pad = int((input_dims - output_dims)/2)
    print(pad)
    
    ###create padding for north pole
# =============================================================================
#     fct_shift = fct.pad(pad_width = {'longitude' : (0,120)}, mode = 'wrap').shift({'longitude' : 120}).isel(longitude = slice(120, 120 + int(360/1.5)))
#     fct_shift_pad = fct_shift.pad(pad_width = {'latitude' : (pad,0)}, mode = 'reflect')
#     shift_pad = fct_shift_pad.isel(latitude = slice (0,pad))
# =============================================================================
    
    ###create padding for north pole and south pole
    fct_shift = fct.pad(pad_width = {'longitude' : (0,120)}, mode = 'wrap').shift({'longitude' : 120}).isel(longitude = slice(120, 120 + int(360/1.5)))
    fct_shift_pad = fct_shift.pad(pad_width = {'latitude' : (pad,2*pad)}, mode = 'reflect')
    shift_pad_north = fct_shift_pad.isel(latitude = slice (0,pad))
    shift_pad_south = fct_shift_pad.isel(latitude = slice (157-2*pad, 157))
    
    
    ###add pole padding to ftc_train
    fct_lat_pad = xr.concat([shift_pad_north,  fct, shift_pad_south], dim = 'latitude')#fct_lat_pad = xr.concat([shift_pad,  fct], dim = 'latitude')
    
    ### pad in the east-west direction
    fct_padded = fct_lat_pad.pad(pad_width = {'longitude' : (pad,pad)}, mode = 'wrap')
    #plt.figure()
    #plt.imshow(hind_padded.isel(forecast_time = 0).lower_t2m.values)
    return fct_padded

def slide_predict(fct_padded, input_dims, output_dims, cnn, basis, clim_probs):
    """"slide the model that was trained on one specific patch over the earth to create predictions everywhere between 90N and 60S
    
    Parameters:
    ----------
    fct_padded: (xarray DataSet) preprocessed forecasts with padding
    input_dims: (int)
    output_dims: (int)
    basis: (numpy array) contains weights of basis functions for the output domain
    clim_probs: ()
    
    Returns:
    -------
    prediction: (numpy array) global prediction 
    
    """
    #initialize 1/3 matrix to save predictions
    prediction = np.ones(shape = (len(fct_padded.forecast_time), int(360/1.5), int(180/1.5) + 10,3))*1/3
    
    #iterate over global and create predictions patchwise
    for lat_i in range(0,int(180/1.5) + 1,output_dims):#range(0,int(180/1.5),output_dims):#range(int(150/1.5),int(180/1.5)
        print(lat_i)
        #patch_lat = fct_predict.isel(latitude = slice(lat_i, lat_i + input_dims))
        for lon_i in range(0,int(360/1.5), output_dims):#range(0,int(30/1.5), output_dims):#
            print(lon_i)
            patch = fct_padded.isel(longitude = slice(lon_i, lon_i + input_dims), latitude = slice(lat_i, lat_i + input_dims))
            #print(patch.sizes)
            
            preds = cnn.predict([patch.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                           np.repeat(basis[np.newaxis,:,:],len(patch.forecast_time),axis=0),
                                           np.repeat(clim_probs[np.newaxis,:,:],len(patch.forecast_time),axis=0)]
                        ).squeeze()
        
            preds = Reshape((output_dims, output_dims,3))(preds)#len(lons),len(lats)
            #preds = tf.transpose(preds, [0,3,1,2])
            prediction[:,lon_i:(lon_i + output_dims), lat_i:(lat_i + output_dims), :] = preds
    prediction = prediction[:,:,0:int(180/1.5) + 1,:]

    return prediction  

def add_coords(pred, fct_ready_to_predict, global_lats, global_lons, lead_output_coords):
    da = xr.DataArray(
                    tf.transpose(pred, [0,3,1,2]),
                    dims=['forecast_time', 'category','longitude', 'latitude'],
                    coords={'forecast_time': fct_ready_to_predict.forecast_time, 'category' : ['below normal', 'near normal', 'above normal'],#verif_train.category, 
                            'latitude': global_lats,
                            'longitude': global_lons
                            }
                )
    da = da.transpose('category','forecast_time','latitude',...)
    da = da.assign_coords(lead_time=lead_output_coords)
    return da

def _create_predictions(model, dg, lead, lons, lats, fct_valid, verif_train, lead_output):
    """Create non-iterative predictions
    returns: prediction in the shape of the input arguments to DataGenerator classe
    """
    
    preds = model.predict(dg).squeeze()
    
    preds = Reshape((len(lons),len(lats),3))(preds)
    preds = tf.transpose(preds, [0,3,1,2])
    
    #transform to original shape of input
    #if dg.verif_dataset:
    #    da = xr.DataArray(
    #                preds,
    #                dims=['forecast_time', 'latitude', 'longitude','variable'],
    #                coords={'forecast_time': fct_valid.forecast_time, 'latitude': fct_train.latitude,
    #                        'longitude': fct_train.longitude},
    #            ).to_dataset() # doesnt work yet
    #else:
    da = xr.DataArray(
                preds,
                dims=['forecast_time', 'category','longitude', 'latitude'],
                coords={'forecast_time': fct_valid.forecast_time, 'category' : verif_train.category, 'latitude': verif_train.latitude,
                        'longitude': verif_train.longitude}
            )
    da = da.transpose('forecast_time','category','latitude',...)
    da = da.assign_coords(lead_time=lead_output)
    return da


def add_valid_time_single(forecast,  lead_output, init_dim='forecast_time'):
    """Creates valid_time(forecast_time, lead_time) for a single lead time and variable
    
    lead_time: pd.Timedelta
    forecast_time: datetime
    """
    
    times = xr.DataArray(
                forecast[init_dim] + lead_output,
                dims=init_dim,
                coords={init_dim: forecast[init_dim]},
            )
            
    forecast = forecast.assign_coords(valid_time=times)
    return forecast

def single_prediction(cnn, dg, lead, lons, lats, fct_valid, verif_train, lead_output):
    """prediction for one var and one lead-time
    
    args:
    time: time slice
    
    """

    preds_test = _create_predictions(cnn, dg, lead, lons, lats, fct_valid, verif_train, lead_output)
    
    # add valid_time coord
    ###preds_test = add_valid_time_from_forecast_reference_time_and_lead_time(preds_test)# only works for complete output
    preds_test = add_valid_time_single(preds_test, lead_output)
    #preds_test = preds_test.to_dataset(name=v)
                                    
    return preds_test



def skill_by_year_single(prediction, terciled_obs, v):
    """version of skill_by_year adjusted to one var and one lead time and flexibel validation period"""
    fct_p = prediction
    obs_p = terciled_obs


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

    # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/7

    # penalize
    penalize = obs_p.where(fct_p!=1, other=-10).mean('category')
    rpss = rpss.where(penalize!=0, other=-10)
    print('number of missing predictions: ', penalize.where(penalize == -10).count().values)

    # clip
    rpss = rpss.clip(-10, 1)

    # average over all forecasts
    rpss_year = rpss.groupby('forecast_time.year').mean()

    # weighted area mean
    weights = np.cos(np.deg2rad(np.abs(rpss_year.latitude)))
    # spatially weighted score averaged over lead_times and variables to one single value
    scores = rpss_year.sel(latitude=slice(None, -60)).weighted(weights).mean('latitude').mean('longitude')
    #scores = scores.to_array().mean(['lead_time', 'variable'])

    return scores.to_dataframe('RPSS') 



def my_loss_rps(y_true, y_pred):
    # - rps as loss, yields almost only pixels with prob = 0 or 1. too sharp forecast, even worse than raw ensemble.
    #https://courses.cs.duke.edu//spring17/compsci590.2/Gneiting2007jasa.pdf
    #https://keras.io/api/losses/#creating-custom-losses
    y_true_cum = tf.cumsum(y_true, axis = -1)
    y_pred_cum = tf.cumsum(y_pred, axis = -1)
    #rps = tf.reduce_sum(tf.square(y_true_cum - y_pred_cum), axis=-1)
    return tf.reduce_sum(tf.square(y_true_cum - y_pred_cum), axis=-1)#rps  # Note the `axis=-1`