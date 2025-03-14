import os
import sys
import json
import numpy as np
import xarray as xr
from tqdm import tqdm

sys.path.append('/lustre/gmeteo/WORK/abadj/deep4downscaling')
import deep4downscaling.viz as viz
import deep4downscaling.trans as trans

sys.path.append('/lustre/gmeteo/WORK/abadj/DL-based_infilling')
import src.data_utils as data_utils

def get_nearest_stations(lat, lon,
                         data_to_extract, var_name,
                         num_neighbours):

    # Get target point coordinates
    target_point = np.array([lat, lon])
    
    # Get all coordinates based on dimensions
    if 'station' in data_to_extract.dims:
        # Case 1: Station dimension
        station_lats = data_to_extract['lat'].values
        station_lons = data_to_extract['lon'].values
        points = np.column_stack((station_lats, station_lons))
        
        # Calculate distances
        distances = np.sqrt(
            (station_lats - target_point[0])**2 + 
            (station_lons - target_point[1])**2
        )
        
        # Get indices and values, excluding NaN values
        valid_data = ~np.isnan(data_to_extract[var_name]).any(dim='time' if 'time' in data_to_extract.dims else None)
        valid_indices = np.where(valid_data)[0]
        valid_distances = distances[valid_indices]
        
        # Sort valid distances and get nearest indices
        nearest_valid_idx = np.argsort(valid_distances)[:num_neighbours]
        nearest_indices = valid_indices[nearest_valid_idx]
        nearest_stations = data_to_extract[var_name].isel(station=nearest_indices)
        
    else:
        # Case 2: Lat/lon dimensions
        lats = data_to_extract['lat'].values
        lons = data_to_extract['lon'].values
        
        # Create meshgrid of coordinates
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        
        # Calculate distances to all points
        distances = np.sqrt(
            (points[:,0] - target_point[0])**2 + 
            (points[:,1] - target_point[1])**2
        )
        
        # Get valid data mask
        if 'time' in data_to_extract.dims:
            valid_data = ~np.isnan(data_to_extract[var_name]).any(dim='time')
        else:
            valid_data = ~np.isnan(data_to_extract[var_name])
        valid_data_flat = valid_data.values.ravel()
        
        # Filter distances for valid data points only
        valid_distances = distances[valid_data_flat]
        valid_indices = np.where(valid_data_flat)[0]
        
        # Get nearest indices among valid points
        nearest_valid_idx = np.argsort(valid_distances)[:num_neighbours]
        nearest_indices = valid_indices[nearest_valid_idx]
        
        # Convert to 2D indices
        lat_indices = nearest_indices // len(lons)
        lon_indices = nearest_indices % len(lons)
        
        # Get values based on dimensions
        if 'time' in data_to_extract.dims:
            nearest_stations = data_to_extract[var_name].isel(
                lat=xr.DataArray(lat_indices),
                lon=xr.DataArray(lon_indices)
            )
        else:
            nearest_stations = data_to_extract[var_name].isel(
                lat=lat_indices,
                lon=lon_indices
            )
            
    return nearest_stations

if __name__ == '__main__':

    # Load paths
    paths = json.load(open('/lustre/gmeteo/WORK/abadj/DL-based_infilling/config.json'))
    data_files_path = f'{paths['data']}/data_files'
    data_proc_path = f'{paths['data']}/data_proc'
    figs_path = paths['figs']

    # Monthly aggregation of the stations data
    stations = xr.open_dataset(f'{data_files_path}/PENINSULAYBALEARES_tasmax_19750101-20201231.nc')
    stations = stations.drop_vars(['alt', 'projection'])
    stations_monthly = stations.resample(time='ME').mean()

    # Monthly aggregation of the target grid data
    target_grid = xr.open_dataset(f'{data_files_path}/tasmax_AEMET.nc')
    target_grid_monthly = target_grid.resample(time='ME').mean()

    # Intersect in time
    target_grid_monthly, stations_monthly = trans.align_datasets(target_grid_monthly, stations_monthly, coord='time')

    # Interpolate co-variables to target_grid
    data_utils.interpolate_variable(input_path=f'{data_files_path}/orog_1km_Spain_FX_v01.nc4',
                                    ref_data_path=f'{data_files_path}/tasmax_AEMET.nc',
                                    output_path=f'{data_proc_path}/orog_5km.nc')
    orog = xr.open_dataset(f'{data_proc_path}/orog_5km.nc')

    data_utils.interpolate_variable(input_path=f'{data_files_path}/distCoast_1km_Spain_FX_v01.nc4',
                                    ref_data_path=f'{data_files_path}/tasmax_AEMET.nc',
                                    output_path=f'{data_proc_path}/distCoast_5km.nc')
    dist_coast = xr.open_dataset(f'{data_proc_path}/distCoast_5km.nc')

    # Set number of closest neighbours (for the stations)
    num_neighbours = 3

    # Iterate over the gridpoints of the target grid data and create the dataset
    rows_data = [] # Here I save the data
    rows_date = [] # Here I save the corresponding dates

    for lat in tqdm(target_grid_monthly['lat'].values, desc='Processing latitudes'):
        for lon in tqdm(target_grid_monthly['lon'].values, desc='Processing longitudes', leave=False):

            # We only get data from non-nan grid-points
            data_spatial = target_grid_monthly.sel(lat=lat).sel(lon=lon)
            data_value = data_spatial['tasmax'].values
            if bool(np.isnan(data_value).any()):
                pass
            else:

                # For indexing
                date = data_spatial['time'].values

                # Extract the closest stations
                nearest_stations = get_nearest_stations(lat=lat, lon=lon,
                                                        data_to_extract=stations_monthly,
                                                        var_name='tasmax', num_neighbours=num_neighbours)

                # Extract the closest orography
                nearest_orog = get_nearest_stations(lat=lat, lon=lon,
                                                    data_to_extract=orog,
                                                    var_name='orog', num_neighbours=1)

                # Extract the closest distance to coast
                nearest_dist_coast = get_nearest_stations(lat=lat, lon=lon,
                                                        data_to_extract=dist_coast,
                                                        var_name='distCoast', num_neighbours=1)

                # Replicate the co-variables across time axis and append to nearest_stations
                # Get base sample from nearest stations
                sample = nearest_stations.values

                # Create arrays for each feature with consistent shape
                num_timesteps = sample.shape[0]
                feature_shape = (num_timesteps, 1)

                # Prepare features
                features = {
                    'date': np.expand_dims(date, axis=1),
                    'orog': np.full(feature_shape, nearest_orog.values.item()),
                    'dist_coast': np.full(feature_shape, nearest_dist_coast.values.item()),
                    'lat': np.full(feature_shape, lat),
                    'lon': np.full(feature_shape, lon),
                    'target': np.expand_dims(data_value, axis=1)
                }

                # Concatenate all features with the base sample
                sample = np.concatenate([sample, features['orog'], features['dist_coast'], features['lat'], features['lon'], features['target']], axis=1)
                print(sample.shape)
                print(features['date'].shape)
                rows_data.append(sample)

                # Save the corresponding date
                rows_date.append(features['date'])

    # Concatenate into a single array
    # Dims: Num_neighbors (Stations), orog, dist_coast, lat, lon, target (grid)
    data_pretraining = np.concatenate(rows_data)
    np.save(f'{data_proc_path}/data_pretraining.npy', data_pretraining)

    # Dims: Date
    dates_pretraining = np.concatenate(rows_date)
    np.save(f'{data_proc_path}/dates_pretraining.npy', dates_pretraining)