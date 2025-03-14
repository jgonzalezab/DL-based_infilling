import sys
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append('/lustre/gmeteo/WORK/abadj/deep4downscaling')
import deep4downscaling.metrics as metrics
import deep4downscaling.trans as trans

# Load paths
paths = json.load(open('/lustre/gmeteo/WORK/abadj/DL-based_infilling/config.json'))
data_files_path = f'{paths['data']}/data_files'
data_proc_path = f'{paths['data']}/data_proc'
preds_path = f'{paths['data']}/preds'
figs_path = paths['figs']
models_path = paths['models']

# Some config
var_name = 'tasmax'

# Target data
target_grid = xr.open_dataset(f'{data_files_path}/tasmax_AEMET.nc')
target_grid_monthly = target_grid.resample(time='ME').mean()

# Prediction data
model_name = 'deep_feedforward'
predictions_test = xr.open_dataset(f'{preds_path}/preds_test_{model_name}.nc')

# Align datasets in time
target_grid_monthly, predictions_test = trans.align_datasets(target_grid_monthly, predictions_test,
                                                           coord='time')

# Define metrics to compute and plot
metrics_to_plot = {
    'RMSE': (metrics.rmse, {'vmin': 0, 'vmax': 2, 'cmap': 'Reds'}, {}),
    'Bias P02': (metrics.bias_quantile, {'vmin': -1.5, 'vmax': 1.5, 'cmap': 'RdBu_r'}, {'quantile': 0.02}),
    'Bias Mean': (metrics.bias_mean, {'vmin': -1.5, 'vmax': 1.5, 'cmap': 'RdBu_r'}, {}),
    'Bias P98': (metrics.bias_quantile, {'vmin': -1.5, 'vmax': 1.5, 'cmap': 'RdBu_r'}, {'quantile': 0.98}),
    'Correlation': (metrics.corr, {'vmin': 0.95, 'vmax': 1, 'cmap': 'YlOrRd'}, {'corr_type': 'pearson', 'deseasonal': False})
}

# Create figure with subplots
fig = plt.figure(figsize=(20, 10))
projection = ccrs.PlateCarree()

for idx, (metric_name, (metric_func, plot_params, metric_params)) in enumerate(metrics_to_plot.items(), 1):
    # Compute metric
    if metric_name in ('Correlation', 'Bias P02', 'Bias P98'):
        metric_kwargs = plot_params.pop('metric_kwargs', {})
        metric_result = metric_func(target=target_grid_monthly, pred=predictions_test,
                                    var_target=var_name, **metric_params)
    else:
        metric_result = metric_func(target=target_grid_monthly, pred=predictions_test, var_target=var_name)
    
    # Create subplot
    ax = fig.add_subplot(1, len(metrics_to_plot), idx, projection=projection)
    
    # Plot metric
    im = plt.pcolormesh(metric_result.coords['lon'].values, metric_result.coords['lat'].values, metric_result[var_name],
                        transform=ccrs.PlateCarree(),
                        **{k: v for k, v in plot_params.items()})
    
    # Add coastlines and borders
    ax.coastlines(resolution='10m')
    
    # Set title
    ax.set_title(metric_name)

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f'{figs_path}/spatial_metrics_{model_name}.pdf', dpi=300, bbox_inches='tight')
plt.close()