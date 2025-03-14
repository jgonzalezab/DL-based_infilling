import sys
import gc
import json
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm 

sys.path.append('/lustre/gmeteo/WORK/abadj/deep4downscaling')
import deep4downscaling.trans as trans
import deep4downscaling.viz as viz
import deep4downscaling.deep.utils as deep_utils
import deep4downscaling.deep.models as deep_models
import deep4downscaling.deep.loss as deep_loss
import deep4downscaling.deep.train as deep_train

sys.path.append('/lustre/gmeteo/WORK/abadj/DL-based_infilling')
import src.data_utils as data_utils
import src.models as models
import src.utils as utils

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# Load paths
paths = json.load(open('/lustre/gmeteo/WORK/abadj/DL-based_infilling/config.json'))
data_files_path = f'{paths['data']}/data_files'
data_proc_path = f'{paths['data']}/data_proc'
preds_path = f'{paths['data']}/preds'
figs_path = paths['figs']
models_path = paths['models']

# Load dataset
data = np.load(f'{data_proc_path}/data_1km.npy')
dates = np.load(f'{data_proc_path}/dates_1km.npy')

# Filter out nans coming from the preallocation of the array
data = data[~np.isnan(data).any(axis=1)]
dates = dates[~np.isnan(dates).any(axis=1)] # It seems that the formatting for the dates is broken the 

######### Select a set of days to interpolat ###########
# Extract unique days from dates
unique_days = np.unique(dates.astype('datetime64[D]'))

# Choose n random days from unique_days
n = 10  # Specify the number of random days to extract
np.random.seed(42)  # For reproducibility
random_days = np.random.choice(unique_days, size=n, replace=False)

# Create masks for each set based on the random days
data_mask = np.isin(dates.astype('datetime64[D]'), random_days)

# Split data according to masks
data_masked = data[data_mask[:, 0]]

# Split dates according to masks (optional, if you need to keep track of dates)
dates_masked = dates[data_mask[:, 0]]

# Split lat and lon according to masks (optional, if you need to keep track of dates)
data_coords = data[data_mask[:, 0], 5:(6+1)]

# Free memory
del data; del dates; gc.collect()
##########################################################################

####### Load and process training data to replicate the processing #######
# Load dataset
data = np.load(f'{data_proc_path}/data_pretraining.npy')
dates = np.load(f'{data_proc_path}/dates_pretraining.npy')

# Extract unique days from dates
unique_days = np.unique(dates.astype('datetime64[D]'))

# Split days into train, validation, and test sets
np.random.seed(42)  # For reproducibility
np.random.shuffle(unique_days)

# Define split ratios (e.g., 70% train, 15% validation, 15% test)
train_ratio, val_ratio = 0.7, 0.15
train_days = unique_days[:int(len(unique_days) * train_ratio)]

# Create masks for each set
train_mask = np.isin(dates.astype('datetime64[D]'), train_days)

# Split data according to masks
train_data = data[train_mask[:, 0]]

# Split dates according to masks (optional, if you need to keep track of dates)
train_dates = dates[train_mask]

# Split lat and lon according to masks (optional, if you need to keep track of dates)
train_coords = data[train_mask[:, 0], 5:(6+1)]
##########################################################################

# Standardize station data
num_neighbours = 3
mean_stations = np.mean(train_data[:, :num_neighbours])
sd_stations = np.std(train_data[:, :num_neighbours])

data_masked[:, :num_neighbours] = (data_masked[:, :num_neighbours] - mean_stations) / sd_stations

# Normalize the co-variables
min_covariables = np.min(train_data[:, num_neighbours:-1], axis=0)
max_covariables = np.max(train_data[:, num_neighbours:-1], axis=0)

data_masked[:, num_neighbours:] = (data_masked[:, num_neighbours:] - min_covariables) / (max_covariables - min_covariables)

# Create dataset and dataloaders
batch_size = 64
empty_output = np.empty_like(data_masked[:, 0, np.newaxis])

dataset = data_utils.MyDataset(inputs=data_masked,
                               outputs=empty_output)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False)

# Load model
model_name = 'deep_feedforward'
model = models.DeepFeedForwardNN(input_size=data_masked.shape[1])

model.load_state_dict(torch.load(f'{models_path}/{model_name}.pt'))
model.eval()

# Compute predictions for the test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictions_test = []
model.to(device)

with torch.no_grad():
    for inputs, _ in tqdm(dataloader):  
        inputs = inputs.to(device)  
        outputs = model(inputs)  
        predictions_test.append(outputs.cpu().numpy()) 

predictions_test = np.concatenate(predictions_test, axis=0)

################################## ÑAPA ALERT ######################################
unique_values = np.unique(dates_masked)

# Generate random dates
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-12-31')
random_dates = pd.date_range(start=start_date, end=end_date, freq='D').to_numpy()

# Map unique values to random dates
value_to_date = {value: random_dates[i] for i, value in enumerate(unique_values)}

# Replace values in the original array with corresponding random dates
dates_masked_random_dates = np.vectorize(value_to_date.get)(dates_masked)
####################################################################################

# After generating predictions_test
template = xr.open_dataset(f'{data_proc_path}/ref_tasmax_AEMET_1km.nc').load()
predictions_test_ds = utils.predictions_to_xarray(predictions=predictions_test, dates=dates_masked_random_dates, coords=data_coords,
                                                  template=template)
predictions_test_ds.to_netcdf(f'{preds_path}/preds_1km_random_dates_{model_name}.nc') # [!!!!!!!] Dates do not correspond to the data due to the ÑAPA [!!!!!!!]