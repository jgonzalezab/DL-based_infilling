import os
import sys
import json
from tqdm import tqdm
import xarray as xr
import xesmf as xe

sys.path.append('/home/jose/Desktop/deep4downscaling')
import deep4downscaling.viz as viz
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/home/jose/Desktop/DL-based_infilling/config.json'))
data_files_path = f'{paths['data']}/data_files'
data_proc_path = f'{paths['data']}/data_proc'
figs_path = paths['figs']

def interpolate_variable(input_path, ref_data_path, output_path):

    os.system(f'cdo griddes {ref_data_path} > {data_proc_path}/grid.txt')
    os.system(f'cdo remapbil,{data_proc_path}/grid.txt {input_path} {output_path}')
    os.system(f'rm {data_proc_path}/grid.txt')

def interpolate_stations(input_path, ref_data_path, output_path):

    import numpy as np
    from scipy.spatial import cKDTree

    ds_unstruct = xr.open_dataset(input_path)
    ds_struct = xr.open_dataset(ref_data_path)

    ds_unstruct, ds_struct = trans.align_datasets(ds_unstruct, ds_struct, coord='time')

    # Load the unstructured dataset.
    # Assumes dimensions: ('time', 'point') for the variable 'data'
    # and coordinate variables 'lat' and 'lon' that are 1D arrays (for points).
    data_unstruct = ds_unstruct['tasmax'].values      # shape: (nt, n_points)
    lats_unstruct = ds_unstruct['lat'].values         # shape: (n_points,)
    lons_unstruct = ds_unstruct['lon'].values         # shape: (n_points,)

    # Load the structured dataset.
    # Assumes dimensions: ('time', 'lat', 'lon') for existing data or grid.
    lat_struct = ds_struct['lat'].values             # e.g., 1D array of latitudes
    lon_struct = ds_struct['lon'].values             # e.g., 1D array of longitudes

    # Create the 2D grid (only spatial dimensions).
    lon2d, lat2d = np.meshgrid(lon_struct, lat_struct)
    grid_points = np.column_stack((lat2d.ravel(), lon2d.ravel()))

    # Build the KDTree for the structured grid points.
    tree = cKDTree(grid_points)

    # Prepare the unstructured points (assumed static over time).
    points_unstruct = np.column_stack((lats_unstruct, lons_unstruct))

    # Compute the mapping indices (only once if spatial locations are static).
    _, indices = tree.query(points_unstruct)

    # Determine the number of time steps.
    nt = data_unstruct.shape[0]
    grid_shape = lat2d.shape  # (n_lat, n_lon)

    # Preallocate an array to hold the mapped values for each time.
    mapped_time_series = np.empty((nt, grid_shape[0], grid_shape[1]))

    # Loop over time to map the unstructured data into the structured grid.
    for t in tqdm(range(nt)):
        # Option: average values if multiple points map to the same grid cell.
        accumulated = np.zeros(lat2d.size)
        counts = np.zeros(lat2d.size)
        # Loop over each unstructured point.
        for i, idx in enumerate(indices):
            accumulated[idx] += data_unstruct[t, i]
            counts[idx] += 1
        # Compute the average for grid cells (avoid division by zero).
        mapped = (accumulated / np.maximum(counts, 1)).reshape(grid_shape)
        mapped_time_series[t] = mapped

    # Create a new xr.Dataset to hold the mapped data, using ds_struct as a reference.
    ds_mapped = xr.Dataset(
        {
            'tasmax': (('time', 'lat', 'lon'), mapped_time_series)
        },
        coords={
            'time': ds_struct['time'].values,
            'lat': ds_struct['lat'].values,
            'lon': ds_struct['lon'].values
        },
        attrs=ds_struct.attrs  # Copy attributes from the structured dataset
    )

    ds_mapped.to_netcdf(output_path)


if __name__ == '__main__':

    # Interpolate dist_coast
    input_path = f'{data_files_path}/distCoast_1km_Spain_FX_v01.nc4'
    ref_data_path = f'{data_files_path}/tasmax_AEMET.nc'
    output_path = f'{data_proc_path}/dist_coast_5km.nc'
    interpolate_variable(input_path=input_path, ref_data_path=ref_data_path, output_path=output_path)

    # Interpolate orog
    input_path = f'{data_files_path}/orog_1km_Spain_FX_v01.nc4'
    ref_data_path = f'{data_files_path}/tasmax_AEMET.nc'
    output_path = f'{data_proc_path}/orog_5km.nc'
    interpolate_variable(input_path=input_path, ref_data_path=ref_data_path, output_path=output_path)

    # Interpolate stations
    input_path = f'{data_files_path}/PENINSULAYBALEARES_tasmax_19750101-20201231.nc'
    ref_data_path = f'{data_files_path}/tasmax_AEMET.nc'
    output_path = f'{data_proc_path}/tasmax_stations_5km.nc'
    interpolate_stations(input_path=input_path, ref_data_path=ref_data_path, output_path=output_path)    

    viz.simple_map_plot(data=ds_mapped.isel(time=0), var_to_plot='tasmax', output_path='./check.pdf')