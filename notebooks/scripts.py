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
    ds_p = xr.concat([bn, n, an],'category').assign_coords(category=['below normal', 'normal', 'above normal'])
    if mask is not None:
        ds_p = ds_p.where(mask)
    if 'tp' in ds_p.data_vars:
        # mask arid grid cells where category_edge are too close to 0
        # we are using a dry mask as in https://doi.org/10.1175/MWR-D-17-0092.1
        tp_arid_mask = tercile_edges.tp.isel(category_edge=0, lead_time=0, drop=True) > 0.01
        ds_p['tp'] = ds_p['tp'].where(tp_arid_mask)
    # ds_p = ds_p * 100 # in percent %
    return ds_p


def print_RPS_per_year(preds):
    """Returns pd.Dataframe of RPS per year."""
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
    
    ## RPSS
    # rps_ML
    rps_ML = xs.rps(obs_p, fct_p, category_edges=None, dim=[], input_distributions='p').compute()
    rps_ML = rps_ML.groupby('forecast_time.year').mean()
    
    rpss = rps_ML
    
    # cleaning
    # check for -inf grid cells
    if (rpss==-np.inf).to_array().any():
        (rpss == rpss.min()).sum()

        # dirty fix
        rpss = rpss.clip(-1, 1)

    mask = xr.ones_like(rpss.isel(lead_time=0, drop=True)).reset_coords(drop=True).t2m
    boundary_tropics = 30
    mask = xr.concat([mask.where(mask.latitude > boundary_tropics),
                    mask.where(np.abs(mask.latitude) <= boundary_tropics),
                    mask.where((mask.latitude < -boundary_tropics) & (mask.latitude > -60))],'area')
    mask = mask.assign_coords(area=['northern_extratropics', 'tropics', 'southern_extratropics'])
    mask.name = 'area'

    mask = mask.where(rpss.t2m.isel(lead_time=0, drop=True).notnull())
    
    # weighted area mean
    weights = np.cos(np.deg2rad(np.abs(mask.latitude)))
    scores = (rpss*mask).weighted(weights).mean('latitude').mean('longitude')
    pd_scores = scores.reset_coords(drop=True).to_dataframe().unstack(0).round(2)
    
    # final score
    scores = rpss.weighted(weights).mean('latitude').mean('longitude')
    # spatially weighted score averaged over lead_times and variables to one single value
    scores = scores.to_array().mean(['lead_time', 'variable'])
    return scores.to_dataframe('RPS')


def assert_predictions_2020(preds_test):
    """Check the variables, coordinates and dimensions of 2020 predictions."""
    # is dataset
    assert isinstance(preds_test, xr.Dataset)

    # has both vars: tp and t2m
    assert 'tp' in preds_test.data_vars
    assert 't2m' in preds_test.data_vars
    
    ## coords
    # forecast_time
    d = pd.date_range(start='2020-01-02', freq='7D', periods=53)
    forecast_time = xr.DataArray(d, dims='forecast_time', coords={'forecast_time':d})
    assert (forecast_time == preds_test['forecast_time']).all()

    # longitude
    lon = np.arange(0., 360., 1.5)
    longitude = xr.DataArray(lon, dims='longitude', coords={'longitude': lon})
    assert (longitude == preds_test['longitude']).all()
    
    # latitude
    lat = np.arange(90., -90.1, 1.5)
    latitude = xr.DataArray(lat, dims='latitude', coords={'latitude': lat})
    assert (latitude == preds_test['latitude']).all()
    
    # lead_time
    lead = [pd.Timedelta(f'{i} d') for i in [14, 28]]
    lead_time = xr.DataArray(lead, dims='lead_time', coords={'lead_time': lead})
    assert (lead_time == preds_test['lead_time']).all()
    
    # category
    cat = np.array(['below normal', 'normal', 'above normal'], dtype='<U12')
    category = xr.DataArray(cat, dims='category', coords={'category': cat})
    assert (category == preds_test['category']).all()
    
    # size
    from dask.utils import format_bytes
    size_in_MB = float(format_bytes(preds_test.nbytes).replace(' MiB',''))
    assert size_in_MB > 50
    assert size_in_MB < 250
    
    # no other dims
    assert set(preds_test.dims) - {'category', 'forecast_time', 'latitude', 'lead_time', 'longitude'} == set()
    