import xarray as xr
import numpy as np
from tqdm import tqdm
from typing import Tuple, Union

def predictions_to_xarray(predictions: np.ndarray, dates: np.ndarray, coords: np.ndarray,
                          template: xr.Dataset) -> xr.Dataset:

    # Get unique dates
    dates_unique = np.unique(dates)

    # Get template data, handling case where there is no time dimension
    if 'time' in template.dims:
        template_data = template.isel(time=0)
    else:
        template_data = template
    
    # Create new dataset with template spatial dims and new time dim
    ds = xr.Dataset(
        data_vars={
            list(template.data_vars.keys())[0]: (
                ['time', 'lat', 'lon'],
                np.full((len(dates_unique), len(template.lat), len(template.lon)), np.nan)
            )
        },
        coords={
            'time': dates_unique.astype('datetime64[D]'),
            'lat': template.lat,
            'lon': template.lon
        }
    )

    # Insert predictions at the exact coordinates
    var_name = list(ds.data_vars.keys())[0]
    for i in tqdm(range(len(predictions)), desc='Inserting predictions'):
        lat, lon = coords[i]
        date = dates[i]
        ds[var_name].loc[dict(time=date, lat=lat, lon=lon)] = predictions[i].item()
    
    return ds
