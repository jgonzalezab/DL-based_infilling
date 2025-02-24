import os
import sys
import json
# from tqdm import tqdm
import xarray as xr
# import xesmf as xe
import numpy as np
# from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset

sys.path.append('/lustre/gmeteo/WORK/abadj/deep4downscaling')
import deep4downscaling.viz as viz
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/lustre/gmeteo/WORK/abadj/DL-based_infilling/config.json'))
data_files_path = f'{paths['data']}/data_files'
data_proc_path = f'{paths['data']}/data_proc'
figs_path = paths['figs']

def interpolate_variable(input_path, ref_data_path, output_path):

    '''
    This function interpolates the input_path grid to the grid from ref_data_path
    and save it as output_path
    '''

    os.system(f'cdo griddes {ref_data_path} > {data_proc_path}/grid.txt')
    os.system(f'cdo remapbil,{data_proc_path}/grid.txt {input_path} {output_path}')
    os.system(f'rm {data_proc_path}/grid.txt')

class PretrainingDataset(Dataset):

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index, :], self.outputs[index, :]

# def interpolate_stations(input_path, ref_data_path, output_path):

#     '''
#     Gets stations data (input_path) and regrids it to the closest lat-long points
#     in the ref_data_path grid, leaving the empty gridpoints as zeros. Finally it 
#     saves it as output_path
#     '''

#     ds_unstruct = xr.open_dataset(input_path)
#     ds_struct = xr.open_dataset(ref_data_path)

#     ds_unstruct, ds_struct = trans.align_datasets(ds_unstruct, ds_struct, coord='time')

#     # Load the unstructured dataset.
#     # Assumes dimensions: ('time', 'point') for the variable 'data'
#     # and coordinate variables 'lat' and 'lon' that are 1D arrays (for points).
#     data_unstruct = ds_unstruct['tasmax'].values      # shape: (nt, n_points)
#     lats_unstruct = ds_unstruct['lat'].values         # shape: (n_points,)
#     lons_unstruct = ds_unstruct['lon'].values         # shape: (n_points,)

#     # Load the structured dataset.
#     # Assumes dimensions: ('time', 'lat', 'lon') for existing data or grid.
#     lat_struct = ds_struct['lat'].values             # e.g., 1D array of latitudes
#     lon_struct = ds_struct['lon'].values             # e.g., 1D array of longitudes

#     # Create the 2D grid (only spatial dimensions).
#     lon2d, lat2d = np.meshgrid(lon_struct, lat_struct)
#     grid_points = np.column_stack((lat2d.ravel(), lon2d.ravel()))

#     # Build the KDTree for the structured grid points.
#     tree = cKDTree(grid_points)

#     # Prepare the unstructured points (assumed static over time).
#     points_unstruct = np.column_stack((lats_unstruct, lons_unstruct))

#     # Compute the mapping indices (only once if spatial locations are static).
#     _, indices = tree.query(points_unstruct)

#     # Determine the number of time steps.
#     nt = data_unstruct.shape[0]
#     grid_shape = lat2d.shape  # (n_lat, n_lon)

#     # Preallocate an array to hold the mapped values for each time.
#     mapped_time_series = np.empty((nt, grid_shape[0], grid_shape[1]))

#     # Loop over time to map the unstructured data into the structured grid.
#     for t in tqdm(range(nt)):
#         # Option: average values if multiple points map to the same grid cell.
#         accumulated = np.nans(lat2d.size)
#         counts = np.zeros(lat2d.size)
#         # Loop over each unstructured point.
#         for i, idx in enumerate(indices):
#             accumulated[idx] += data_unstruct[t, i]
#             counts[idx] += 1
#         # Compute the average for grid cells (avoid division by zero).
#         mapped = (accumulated / np.maximum(counts, 1)).reshape(grid_shape)
#         mapped_time_series[t] = mapped

#     # Create a new xr.Dataset to hold the mapped data, using ds_struct as a reference.
#     ds_mapped = xr.Dataset(
#         {
#             'tasmax': (('time', 'lat', 'lon'), mapped_time_series)
#         },
#         coords={
#             'time': ds_struct['time'].values,
#             'lat': ds_struct['lat'].values,
#             'lon': ds_struct['lon'].values
#         },
#         attrs=ds_struct.attrs  # Copy attributes from the structured dataset
#     )

#     ds_mapped.to_netcdf(output_path)
