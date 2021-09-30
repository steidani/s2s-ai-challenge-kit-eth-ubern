# -*- coding: utf-8 -*-
"""
Functions to load and prepare data for training/validating/testing Ml models
"""
import xarray as xr
xr.set_options(display_style='text')


def load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = 'server', var_list = ['tp','t2m','sm20','sst']):
    """
    Parameters
    ----------
    data: string, {'hind_2000-2019', 'obs_2000-2019', 'obs_terciled_2000-2019','obs_tercile_edges_2000-2019'}
    aggregation: string, {'biweekly','weekly'}
    path: string or 2-element list, {'server', 'local_n', user-defined path to the data}, 
                    server assumes that this function is called within a file in the home directory on the compute server
    var_list: list, list of variables
    """
    
    if path == 'local_n':
        cache_path = '../template/data' 
        path_add_vars = '../data'
    elif path == 'server':
        cache_path = '../../Data/s2s_ai/data'
        path_add_vars = '../../Data/s2s_ai'
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
    
    return dat