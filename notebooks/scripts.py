import xarray as xr
import pandas as pd
import numpy as np

cache_path = '../data'


def add_valid_time_from_forecast_reference_time_and_lead_time(forecast, init_dim='forecast_time'):
    """Creates valid_time(forecast_time, lead_time).
    
    lead_time: pd.Timedelta
    forecast_time: datetime
    """
    times = xr.concat(
        [
            xr.DataArray(
                forecast[init_dim] + lead,
                dims=init_dim,
                coords={init_dim: forecast[init_dim]},
            )
            for lead in forecast.lead_time
        ],
        dim="lead_time",
        join="inner",
        compat="broadcast_equals",
    )
    forecast = forecast.assign_coords(valid_time=times)
    return forecast


def make_probabilistic(ds, tercile_edges, member_dim='realization', mask=None):
    """Compute probabilities from ds (observations or forecasts) based on tercile_edges."""
    # broadcast
    if 'forecast_time' not in tercile_edges.dims and 'weekofyear' in tercile_edges.dims:
        tercile_edges = tercile_edges.sel(weekofyear=ds.forecast_time.dt.weekofyear)
    bn = ds < tercile_edges.isel(category_edge=0, drop=True)  # below normal
    n = (ds >= tercile_edges.isel(category_edge=0, drop=True)) & (ds < tercile_edges.isel(category_edge=1, drop=True))  # normal
    an = ds >= tercile_edges.isel(category_edge=1, drop=True)  # above normal
    if member_dim in ds.dims:
        bn = bn.mean(member_dim)
        an = an.mean(member_dim)
        n = n.mean(member_dim)
    ds_p = xr.concat([bn, n, an],'category').assign_coords(category=['below normal', 'near normal', 'above normal'])
    if mask is not None:
        ds_p = ds_p.where(mask)
    if 'tp' in ds_p.data_vars:
        # mask arid grid cells where category_edge are too close to 0
        # we are using a dry mask as in https://doi.org/10.1175/MWR-D-17-0092.1
        tp_arid_mask = tercile_edges.tp.isel(category_edge=0, lead_time=0, drop=True) > 0.01
        ds_p['tp'] = ds_p['tp'].where(tp_arid_mask)
    # ds_p = ds_p * 100 # in percent %
    return ds_p


def skill_by_year(preds):
    """Returns pd.Dataframe of RPSS per year."""
    # similar verification_RPSS.ipynb
    # as scorer bot but returns a score for each year
    import xarray as xr
    import xskillscore as xs
    import pandas as pd
    import numpy as np
    xr.set_options(keep_attrs=True)
    
    # from root
    #renku storage pull data/forecast-like-observations_2020_biweekly_terciled.nc
    #renku storage pull data/hindcast-like-observations_2000-2019_biweekly_terciled.nc
    
    if 2020 in preds.forecast_time.dt.year:
        obs_p = xr.open_dataset(f'{cache_path}/forecast-like-observations_2020_biweekly_terciled.nc').sel(forecast_time=preds.forecast_time)
    else:
        obs_p = xr.open_dataset(f'{cache_path}/hindcast-like-observations_2000-2019_biweekly_terciled.zarr', engine='zarr').sel(forecast_time=preds.forecast_time)

    # ML probabilities
    fct_p = preds
    
    # check inputs
    assert_predictions_2020(obs_p)
    assert_predictions_2020(fct_p)

    
    # climatology
    clim_p = xr.DataArray([1/3, 1/3, 1/3], dims='category', coords={'category':['below normal', 'near normal', 'above normal']}).to_dataset(name='tp')
    clim_p['t2m'] = clim_p['tp']
    
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
    rpss = rpss.where(penalize!=0,other=-10)

    # clip
    rpss = rpss.clip(-10, 1)

    # average over all forecasts
    rpss = rpss.groupby('forecast_time.year').mean()
    
    # weighted area mean
    weights = np.cos(np.deg2rad(np.abs(rpss.latitude)))
    # spatially weighted score averaged over lead_times and variables to one single value
    scores = rpss.sel(latitude=slice(None, -60)).weighted(weights).mean('latitude').mean('longitude')
    scores = scores.to_array().mean(['lead_time', 'variable'])
    return scores.to_dataframe('RPSS')


def assert_predictions_2020(preds_test, exclude='weekofyear'):
    """Check the variables, coordinates and dimensions of 2020 predictions."""
    from xarray.testing import assert_equal # doesnt care about attrs but checks coords
    
    # is dataset
    assert isinstance(preds_test, xr.Dataset)

    # has both vars: tp and t2m
    if 'data_vars' in exclude:
        assert 'tp' in preds_test.data_vars
        assert 't2m' in preds_test.data_vars
    
    ## coords
    # ignore weekofyear coord if not dim
    if 'weekofyear' in exclude and 'weekofyear' in preds_test.coords and 'weekofyear' not in preds_test.dims:
        preds_test = preds_test.drop('weekofyear')
    
    # forecast_time
    if 'forecast_time' in exclude:
        d = pd.date_range(start='2020-01-02', freq='7D', periods=53)
        forecast_time = xr.DataArray(d, dims='forecast_time', coords={'forecast_time':d}, name='forecast_time')
        assert_equal(forecast_time,  preds_test['forecast_time'])

    # longitude
    if 'longitude' in exclude:
        lon = np.arange(0., 360., 1.5)
        longitude = xr.DataArray(lon, dims='longitude', coords={'longitude': lon}, name='longitude')
        assert_equal(longitude, preds_test['longitude'])

    # latitude
    if 'latitude' in exclude:
        lat = np.arange(-90., 90.1, 1.5)[::-1]
        latitude = xr.DataArray(lat, dims='latitude', coords={'latitude': lat}, name='latitude')
        assert_equal(latitude, preds_test['latitude'])
    
    # lead_time
    if 'lead_time' in exclude:
        lead = [pd.Timedelta(f'{i} d') for i in [14, 28]]
        lead_time = xr.DataArray(lead, dims='lead_time', coords={'lead_time': lead}, name='lead_time')
        assert_equal(lead_time, preds_test['lead_time'])
    
    # category
    if 'category' in exclude:
        cat = np.array(['below normal', 'near normal', 'above normal'], dtype='<U12')
        category = xr.DataArray(cat, dims='category', coords={'category': cat}, name='category')
        assert_equal(category, preds_test['category'])
    
    # size
    if 'size' in exclude:
        from dask.utils import format_bytes
        size_in_MB = float(format_bytes(preds_test.nbytes).split(' ')[0])
        # todo: refine for dtypes
        assert size_in_MB > 50
        assert size_in_MB < 250
    
    # no other dims
    if 'dims' in exclude:
        assert set(preds_test.dims) - {'category', 'forecast_time', 'latitude', 'lead_time', 'longitude'} == set()
    
