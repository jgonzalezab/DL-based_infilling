import sys
import json
import xarray as xr
import numpy as np
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
data = np.load(f'{data_proc_path}/data_pretraining.npy')
dates = np.load(f'{data_proc_path}/dates_pretraining.npy')

######### Split into training, valid and test sets based on days ###########
# Extract unique days from dates
unique_days = np.unique(dates.astype('datetime64[D]'))

# Split days into train, validation, and test sets
np.random.seed(42)  # For reproducibility
np.random.shuffle(unique_days)

# Define split ratios (e.g., 70% train, 15% validation, 15% test)
train_ratio, val_ratio = 0.7, 0.15
train_days = unique_days[:int(len(unique_days) * train_ratio)]
val_days = unique_days[int(len(unique_days) * train_ratio):int(len(unique_days) * (train_ratio + val_ratio))]
test_days = unique_days[int(len(unique_days) * (train_ratio + val_ratio)):]

# Create masks for each set
train_mask = np.isin(dates.astype('datetime64[D]'), train_days)
val_mask = np.isin(dates.astype('datetime64[D]'), val_days)
test_mask = np.isin(dates.astype('datetime64[D]'), test_days)

# Split data according to masks
train_data = data[train_mask[:, 0]]
val_data = data[val_mask[:, 0]]
test_data = data[test_mask[:, 0]]

# Split dates according to masks (optional, if you need to keep track of dates)
train_dates = dates[train_mask]
val_dates = dates[val_mask]
test_dates = dates[test_mask]

# Split lat and lon according to masks (optional, if you need to keep track of dates)
train_coords = data[train_mask[:, 0], 5:(6+1)]
val_coords = data[val_mask[:, 0], 5:(6+1)]
test_coords = data[test_mask[:, 0], 5:(6+1)]
##########################################################################

# Standardize station data
num_neighbours = 3
mean_stations = np.mean(train_data[:, :num_neighbours])
sd_stations = np.std(train_data[:, :num_neighbours])

train_data[:, :num_neighbours] = (train_data[:, :num_neighbours] - mean_stations) / sd_stations
val_data[:, :num_neighbours] = (val_data[:, :num_neighbours] - mean_stations) / sd_stations
test_data[:, :num_neighbours] = (test_data[:, :num_neighbours] - mean_stations) / sd_stations

# Normalize the co-variables
min_covariables = np.min(train_data[:, num_neighbours:-1], axis=0)
max_covariables = np.max(train_data[:, num_neighbours:-1], axis=0)

train_data[:, num_neighbours:-1] = (train_data[:, num_neighbours:-1] - min_covariables) / (max_covariables - min_covariables)
val_data[:, num_neighbours:-1] = (val_data[:, num_neighbours:-1] - min_covariables) / (max_covariables - min_covariables)
test_data[:, num_neighbours:-1] = (test_data[:, num_neighbours:-1] - min_covariables) / (max_covariables - min_covariables)

# Create dataset and dataloaders
batch_size = 64

train_dataset = data_utils.MyDataset(inputs=train_data[:, :-1],
                                     outputs=train_data[:, -1, np.newaxis])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

valid_dataset = data_utils.MyDataset(inputs=val_data[:, :-1],
                                     outputs=val_data[:, -1, np.newaxis])
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)

test_dataset = data_utils.MyDataset(inputs=test_data[:, :-1],
                                    outputs=test_data[:, -1, np.newaxis])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

# Set the DL model
model_name = 'deep_feedforward'
model = models.DeepFeedForwardNN(input_size=train_data[:, :-1].shape[1])

# Configure the training of the DL model
num_epochs = 10000
patience_early_stopping = 20

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)

loss_function = deep_loss.MseLoss(ignore_nans=False)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loss, val_loss = deep_train.standard_training_loop(model=model, model_name=model_name, model_path=models_path,
                                                         device=device, num_epochs=num_epochs,
                                                         loss_function=loss_function, optimizer=optimizer,
                                                         train_data=train_dataloader, valid_data=valid_dataloader,
                                                         patience_early_stopping=patience_early_stopping)

# Load model
model_name = 'deep_feedforward'
model = models.DeepFeedForwardNN(input_size=train_data[:, :-1].shape[1])

model.load_state_dict(torch.load(f'{models_path}/{model_name}.pt'))
model.eval()

# Compute predictions for the test set
predictions_test = []
model.to(device)

with torch.no_grad():
    for inputs, _ in tqdm(test_dataloader):  
        inputs = inputs.to(device)  
        outputs = model(inputs)  
        predictions_test.append(outputs.cpu().numpy()) 

predictions_test = np.concatenate(predictions_test, axis=0)

# After generating predictions_test
template = xr.open_dataset(f'{data_files_path}/tasmax_AEMET.nc').load()
predictions_test_ds = utils.predictions_to_xarray(predictions=predictions_test, dates=test_dates, coords=test_coords,
                                                  template=template)
predictions_test_ds.to_netcdf(f'{preds_path}/preds_test_{model_name}.nc')