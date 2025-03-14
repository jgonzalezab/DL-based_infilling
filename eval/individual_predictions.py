import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append('/lustre/gmeteo/WORK/abadj/deep4downscaling')
import deep4downscaling.trans as trans

# Load paths from config
paths = json.load(open('/lustre/gmeteo/WORK/abadj/DL-based_infilling/config.json'))
data_files_path = f'{paths["data"]}/data_files'
data_proc_path = f'{paths["data"]}/data_proc'
preds_path = f'{paths["data"]}/preds'
figs_path = paths['figs']
models_path = paths['models']

# Some config
var_name = 'tasmax'
model_name = 'deep_feedforward'

# Target data
target_grid = xr.open_dataset(f'{data_files_path}/tasmax_AEMET.nc')
target_grid_monthly = target_grid.resample(time='ME').mean()

# Prediction data
predictions_test = xr.open_dataset(f'{preds_path}/preds_test_{model_name}.nc')

# Align datasets in time
target_grid_monthly, predictions_test = trans.align_datasets(target_grid_monthly, predictions_test,
                                                             coord='time')

# Get time steps to plot and their corresponding vmin and vmax values
time_steps = ['1975-02-28', '2005-07-31', '2020-10-31']
vmin_values = [5, 20, 10]  # Example vmin for each time step
vmax_values = [20, 40, 25]  # Example vmax for each time step

# Create figure with subplots (2 rows: target and prediction)
fig = plt.figure(figsize=(20, 12))
projection = ccrs.PlateCarree()

for idx, time_step in enumerate(time_steps):
    # Plot target data
    ax1 = fig.add_subplot(2, len(time_steps), idx + 1, projection=projection)
    target_slice = target_grid_monthly.sel(time=time_step)[var_name]
    
    im1 = plt.pcolormesh(target_slice.coords['lon'].values, 
                         target_slice.coords['lat'].values, 
                         target_slice.values,
                         transform=ccrs.PlateCarree(),
                         cmap='RdBu_r',
                         vmin=vmin_values[idx],
                         vmax=vmax_values[idx])
    ax1.coastlines(resolution='10m')
    ax1.set_title(f'Target - {str(time_step)[:10]}')
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal')
    cbar1.set_label('Temperature (°C)')
    
    # Plot prediction data
    ax2 = fig.add_subplot(2, len(time_steps), idx + len(time_steps) + 1, projection=projection)
    pred_slice = predictions_test.sel(time=time_step)[var_name]
    
    im2 = plt.pcolormesh(pred_slice.coords['lon'].values, 
                         pred_slice.coords['lat'].values, 
                         pred_slice.values,
                         transform=ccrs.PlateCarree(),
                         cmap='RdBu_r',
                         vmin=vmin_values[idx],
                         vmax=vmax_values[idx])
    ax2.coastlines(resolution='10m')
    ax2.set_title(f'Prediction - {str(time_step)[:10]}')
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal')
    cbar2.set_label('Temperature (°C)')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'{figs_path}/individual_predictions_comparison_{model_name}.pdf', dpi=300, bbox_inches='tight')
plt.close()
