import sys
import json
import xarray as xr

sys.path.append('/home/jose/Desktop/deep4downscaling')
import deep4downscaling.viz as viz

# Load paths
paths = json.load(open('/home/jose/Desktop/DL-based_infilling/config.json'))
data_path = paths['data']
figs_path = paths['figs']