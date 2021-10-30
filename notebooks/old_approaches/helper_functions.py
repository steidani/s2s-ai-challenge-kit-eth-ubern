# -*- coding: utf-8 -*-
"""
helper functions for KIT S2S database
"""
import numpy as np
import pandas as pd
import xarray as xr

import os
from glob import glob

from scripts import add_valid_time_from_forecast_reference_time_and_lead_time


def get_filenames(datapath, model, date, file_type):
    """
    Parameters
    ----------
    datapath : string, path to the s2s folder
    model : string, e.g. ecmwf
    date: string, e.g. 2017-01-01
    file_type : string, e.g. TH

    Returns
    -------
    filenames : list of strings
    """
    #https://www.geeksforgeeks.org/python-os-path-join-method/
    
    path =  os.path.join(datapath,model, date[0:4], date[0:10].replace('-', ''))
    
    CF = '{}{}_00_{}'.format(file_type, date[0:10].replace('-', ''),'CF')
    PF19 = '{}{}_00_{}_0[1-9]'.format(file_type, date[0:10].replace('-', ''),'PF')
    PF10 = '{}{}_00_{}_10'.format(file_type, date[0:10].replace('-', ''),'PF')
    
    filenames = glob(os.path.join(path,PF19))
    filenames.append(glob(os.path.join(path,PF10))[0])
    filenames.append(glob(os.path.join(path,CF))[0])
           
    return filenames

def concat_ensemble(filenames, file_type):
    """
    opens h5netcdfs and concats all ensemble members into one dataset
    
    Parameters
    ----------
    filenames : list of strings, [path/filename]
    file_type : string, e.g. TH

    Returns
    -------
    ds : xarray dataset containing ensemble for one forecast initial time and one file_type
    """
    ds_list = []

    ds = xr.open_dataset(filenames[-1:][0], engine = 'h5netcdf')
    ds = ds.assign_coords(realization = 0)
    ds_list.append(ds)

    for filename in filenames[0:-1]:
        #iteratively read the ensembles and add them to ds
        ds = xr.open_dataset(filename, engine = 'h5netcdf')
        ds = ds.assign_coords(realization = np.int64(filename[-2:]))
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim = 'realization')

    ds = ds.rename({'lon':'longitude', 'lat':'latitude'})

    if file_type == 'TH':
        ds = ds.isel(lev = 0 ).reset_coords('lev', drop=True)
    elif file_type == 'SD':
        ds = ds.isel(height = 0 ).reset_coords('height', drop=True)
    elif file_type == 'SDA':
        ds = ds.drop_vars('depth_bnds').isel(height = 0, depth = 0).reset_coords(('height', 'depth'), drop=True)
    elif file_type == 'P':
        ds = ds.sel(plev = 50000).sel(plev_2 = 50000).reset_coords(('plev', 'plev_2'), drop=True)
        
    return ds

def aggregate_weekly(da, s2s = False):
    """
    Aggregate initialized S2S forecasts weekly for xr.DataArrays.
    Use ds.map(aggregate_weekly) for xr.Datasets.
    
    This function does not return values for week 1, 
    since lead_time = 0 is not available for certain variables (e.g. t2m) or is zero (e.g. tp).
    
    Applies to the ECMWF S2S data model: https://confluence.ecmwf.int/display/S2S/Parameters
    
    Parameters
    ----------
    da : xarray dataarray with time coordinate

    Returns
    -------
    da_weekly : xarray dataarray containing weekly aggregated quantities
    """
    if s2s == True: # correct for variables without a value at lead_time = 0?
        da = da.assign_coords({'forecast_time': da.time[0].values})
        da = da.expand_dims('forecast_time')
        da = da.assign_coords({'lead_time': da.time - da.time[0]})
        da = da.swap_dims({'time': 'lead_time'})
    
    w2 = [pd.Timedelta(f'{i} d') for i in range(7,14)]
    w2 = xr.DataArray(w2,dims='lead_time', coords={'lead_time':w2})
    
    w3 = [pd.Timedelta(f'{i} d') for i in range(14,21)]
    w3 = xr.DataArray(w3,dims='lead_time', coords={'lead_time':w3})
    
    w4 = [pd.Timedelta(f'{i} d') for i in range(21,28)]
    w4 = xr.DataArray(w4,dims='lead_time', coords={'lead_time':w4})
    
    w4 = [pd.Timedelta(f'{i} d') for i in range(21,28)]
    w4 = xr.DataArray(w4,dims='lead_time', coords={'lead_time':w4})
    
    w5 = [pd.Timedelta(f'{i} d') for i in range(28,35)]
    w5 = xr.DataArray(w5,dims='lead_time', coords={'lead_time':w5})
    
    w6 = [pd.Timedelta(f'{i} d') for i in range(35,42)]
    w6 = xr.DataArray(w6,dims='lead_time', coords={'lead_time':w6})
    
    
    weekly_lead = [pd.Timedelta(f"{i} d") for i in [7, 14, 21, 28, 35,]] # take first day of weekly average as new coordinate

    v = da.name
    if v in ['tp', 'ttr']:#climetlab_s2s_ai_challenge.CF_CELL_METHODS[v] == 'sum': # weekly difference for sum variables: tp and ttr
        #d1 = da.sel(lead_time=pd.Timedelta("7 d")) - da.sel(lead_time=pd.Timedelta("0 d"))
        d2 = da.sel(lead_time=pd.Timedelta("14 d")) - da.sel(lead_time=pd.Timedelta("7 d"))
        d3 = da.sel(lead_time=pd.Timedelta("21 d")) - da.sel(lead_time=pd.Timedelta("14 d"))
        d4 = da.sel(lead_time=pd.Timedelta("28 d")) - da.sel(lead_time=pd.Timedelta("21 d"))
        d5 = da.sel(lead_time=pd.Timedelta("35 d")) - da.sel(lead_time=pd.Timedelta("28 d"))
        d6 = da.sel(lead_time=pd.Timedelta("42 d")) - da.sel(lead_time=pd.Timedelta("35 d"))
        
    else: # t2m, see climetlab_s2s_ai_challenge.CF_CELL_METHODS # biweekly: mean [day 14, day 27]
        #d1 = da.sel(lead_time=w1).mean('lead_time')
        d2 = da.sel(lead_time=w2).mean('lead_time')
        d3 = da.sel(lead_time=w3).mean('lead_time')
        d4 = da.sel(lead_time=w4).mean('lead_time')
        d5 = da.sel(lead_time=w5).mean('lead_time')
        d6 = da.sel(lead_time=w6).mean('lead_time')
        
    da_weekly = xr.concat([d2,d3,d4,d5,d6],'lead_time').assign_coords(lead_time=weekly_lead)
    
    da_weekly = add_valid_time_from_forecast_reference_time_and_lead_time(da_weekly)
    da_weekly['lead_time'].attrs = {'long_name':'forecast_period', 'description': 'Forecast period is the time interval between the forecast reference time and the validity time.',
                         'aggregate': 'The pd.Timedelta corresponds to the first day of a weekly aggregate.',
                         'week34_t2m': 'mean[day 14, 27]',
                         'week56_t2m': 'mean[day 28, 41]',
                         'week34_tp': 'day 28 minus day 14',
                         'week56_tp': 'day 42 minus day 28'}
    
    return da_weekly

def get_single_forecast(datapath, date, file_type):
    #get filenames
    filenames = get_filenames(datapath = datapath, model = 'ecmwf', date = date, file_type = file_type)

    #open and concat files
    ds = concat_ensemble(filenames, file_type)
    
    #compute weekly aggregates
    ds_w = ds.map(aggregate_weekly, s2s = True)
    
    #ds_w = ds_w.map(ensure_attributes, biweekly=False) this downloads data, so omit this step            
    ds_w = ds_w.sortby('forecast_time')
    
    return ds_w

def replace_unavailable_dates(date_list):
    #in december of all years: 17, 24, 31 are missing, but 18,25 dec and 1 jan of the following year exist (except for 2017)
    date_list = [d.replace('12-17', '12-18') for d in date_list]
    date_list = [d.replace('12-24', '12-25') for d in date_list]
    #date_list = [d.replace('12-31', '01-01') for d in date_list] #year has to change as well
    replacements = {'2000-12-31':'2001-01-01','2001-12-31':'2002-01-01','2002-12-31':'2003-01-01','2003-12-31':'2004-01-01',
                    '2004-12-31':'2005-01-01','2005-12-31':'2006-01-01','2006-12-31':'2007-01-01','2007-12-31':'2008-01-01',
                    '2008-12-31':'2009-01-01','2009-12-31':'2010-01-01','2010-12-31':'2011-01-01','2011-12-31':'2012-01-01',
                    '2012-12-31':'2013-01-01','2013-12-31':'2014-01-01','2014-12-31':'2015-01-01','2015-12-31':'2016-01-01',
                    '2016-12-31':'2017-01-01'}
    replacer = replacements.get  # For faster gets.
    date_list = [replacer(d,d) for d in date_list]

    #more missing values for in jan and feb 2017:
    replacements = {'2017-01-02':'2017-01-04','2017-01-09':'2017-01-08','2017-01-16':'2017-01-15','2017-01-23':'2017-01-22',
                    '2017-01-30':'2017-01-29','2017-02-06':'2017-02-05','2017-02-13':'2017-02-12','2017-02-20':'2017-02-19',
                    '2017-02-27':'2017-02-26'}
    replacer = replacements.get  # For faster gets.
    date_list = [replacer(d,d) for d in date_list]

    #drop dates that are not available: everything after 2017-12-10
    date_list = [d for d in date_list if '2018' not in d]
    date_list = [d for d in date_list if '2019' not in d]
    date_list = [d for d in date_list if d not in ['2017-12-18','2017-12-25','2017-12-31']]
    ##also interesting: https://stackoverflow.com/questions/18819606/python-remove-a-set-of-a-list-from-another-list
    return date_list

def get_multiple_forecasts(datapath, date_list, file_type):
    
    date_list_clean = replace_unavailable_dates(date_list)
    
    ds_weekly_list = []
    for d in date_list_clean:
        print(d)
        ds_weekly = get_single_forecast(datapath, d, file_type)
        ds_weekly_list.append(ds_weekly)
        #maybe better to concat after every tenth dataset
    ds_multiple = xr.concat(ds_weekly_list, dim = 'forecast_time')   
    
    #replace forecast_time and valid_time whenever dates were replaced in the beginning
    ds_multiple.coords['forecast_time'] = np.array(date_list, dtype='datetime64[ns]')
    ds_multiple = add_valid_time_from_forecast_reference_time_and_lead_time(ds_multiple)
    
    return ds_multiple

#pacific easier to work with, but need to transform back for final prediction
def get_longitude_name(ds):
    """Get the name of the longitude dimension by CF unit"""
    lonn = []
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = ds.to_dataset()
    for dimn in ds.dims.keys():
        if ('units' in ds[dimn].attrs and
            ds[dimn].attrs['units'] in ['degree_east', 'degrees_east']):
            lonn.append(dimn)
    if len(lonn) == 1:
        return lonn[0]
    elif len(lonn) > 1:
        errmsg = 'More than one longitude coordinate found by unit.'
    else:
        errmsg = 'Longitude could not be identified by unit.'
    raise ValueError(errmsg)

def antimeridian_pacific(ds, lonn=None):
    """Returns True if the antimeridian is in the Pacific (i.e. longitude runs
    from -180 to 180."""
    if lonn is None:
        lonn = get_longitude_name(ds)
    if ds[lonn].min() < 0 or ds[lonn].max() < 180:
        return True
    return False

def flip_antimeridian(ds, to='Pacific', lonn='lon'):
    """
    #https://git.iac.ethz.ch/utility_functions/utility_functions_python/-/blob/regionmask/xarray.py
    
    Flip the antimeridian (i.e. longitude discontinuity) between Europe
    (i.e., [0, 360)) and the Pacific (i.e., [-180, 180)).
    Parameters:
    - ds (xarray.Dataset or .DataArray): Has to contain a single longitude
      dimension.
    - to='Pacific' (str, optional): Flip antimeridian to one of
      * 'Europe': Longitude will be in [0, 360)
      * 'Pacific': Longitude will be in [-180, 180)
    - lonn=None (str, optional): Name of the longitude dimension. If None it
      will be inferred by the CF convention standard longitude unit.
    Returns:
    same type as input ds
    """

    attrs = ds[lonn].attrs

    if to.lower() == 'europe' and not antimeridian_pacific(ds):
        return ds  # already correct, do nothing
    elif to.lower() == 'pacific' and antimeridian_pacific(ds):
        return ds  # already correct, do nothing
    elif to.lower() == 'europe':
        ds = ds.assign_coords(**{lonn: (ds[lonn] % 360)})
    elif to.lower() == 'pacific':
        ds = ds.assign_coords(**{lonn: (((ds[lonn] + 180) % 360) - 180)})
    else:
        errmsg = 'to has to be one of [Europe | Pacific] not {}'.format(to)
        raise ValueError(errmsg)

    was_da = isinstance(ds, xr.core.dataarray.DataArray)
    if was_da:
        da_varn = ds.name
        if da_varn is None:
            da_varn = 'data'
        enc = ds.encoding
        ds = ds.to_dataset(name=da_varn)

    idx = np.argmin(ds[lonn].data)
    varns = [varn for varn in ds.variables if lonn in ds[varn].dims]
    for varn in varns:
        if xr.__version__ > '0.10.8':
            ds[varn] = ds[varn].roll(**{lonn: -idx}, roll_coords=False)
        else:
            ds[varn] = ds[varn].roll(**{lonn: -idx})

    ds[lonn].attrs = attrs
    if was_da:
        da = ds[da_varn]
        da.encoding = enc
        return da
    return ds


def select_region(ds, area = None):
    """
    Select specific region.
    Parameters
    ----------
    ds : xarray data set
    area : Either 'tropics', 'south-africa' or None. The default is None.
    Returns
    -------
    ds : xarray data set that only contains the data for the specific region.
    
    
    Northern extratropics [90N-30N]: NET
    Tropics (29N-29S): T
    Southern extratropics [30S-60S]: SET
    Europe: EU


    """
    if area =='NET':
        ds = ds.sel(lat = slice(-20,50))
        return ds
    elif area == 'south-africa':
        lc = ds.coords["lon"]
        ds = ds.sel(lat = slice(-50,0)).sel(lon = lc[((lc <=50) | (lc >= (350)))])
        return ds
    elif area == 'large_sa':
        lc = ds.coords["lon"]
        ds = ds.sel(lat = slice(-90,35)).sel(lon = lc[((lc <= 120) | (lc >= (280)))])
        return ds
    else:
        print('NO REGION WAS SELECTED')
        return ds