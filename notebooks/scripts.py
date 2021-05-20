import xarray as xr
import pandas as pd

def add_valid_time_from_forecast_reference_time_and_lead_time(benchmark, init_dim='forecast_time'):
    """Creates valid_time(forecast_time, lead_time).
    
    lead_time: pd.Timedelta
    forecast_time: datetime
    """
    times = xr.concat(
        [
            xr.DataArray(
                benchmark[init_dim] + lead,
                dims=init_dim,
                coords={init_dim: benchmark[init_dim]},
            )
            for lead in benchmark.lead_time
        ],
        dim="lead_time",
        join="inner",
        compat="broadcast_equals",
    )
    benchmark = benchmark.assign_coords(valid_time=times)
    return benchmark



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
    ds_p = xr.concat([bn,n,an],'category').assign_coords(category=['below normal','normal','above normal'])
    if mask is not None:
        ds_p = ds_p.where(mask)
    if 'tp' in ds_p.data_vars:
        # mask arid grid cells where category_edge are too close to 0
        # we are using a dry mask as in https://doi.org/10.1175/MWR-D-17-0092.1
        tp_arid_mask = tercile_edges.tp.isel(category_edge=0, lead_time=0, drop=True) > 0.01
        ds_p['tp'] = ds_p['tp'].where(tp_arid_mask)
    # ds_p = ds_p * 100 # in percent %
    return ds_p