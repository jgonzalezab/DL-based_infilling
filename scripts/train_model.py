import sys
import json
import xarray as xr
import numpy as np

sys.path.append('/lustre/gmeteo/WORK/abadj/deep4downscaling')
import deep4downscaling.trans as trans
import deep4downscaling.viz as viz
import deep4downscaling.deep.utils as deep_utils
import deep4downscaling.deep.models as deep_models
import deep4downscaling.deep.loss as deep_loss
import deep4downscaling.deep.train as deep_train

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

# Load paths
paths = json.load(open('/lustre/gmeteo/WORK/abadj/DL-based_infilling/config.json'))
data_files_path = f'{paths['data']}/data_files'
data_proc_path = f'{paths['data']}/data_proc'
figs_path = paths['figs']

# Load dataset
data = np.load(f'{data_proc_path}/data_pretraining.npy')

# Num observations per month
num_obs_per_month = 100400 # (251*400)
num_obs = data.shape[0] / num_obs_per_month