import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import glob
import urllib.request
import tarfile
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QuantScheme
from utils.imresize import imresize
from utils.models import *
from utils.helpers import *
from utils.downloader import get_tar_path
from utils.inference import load_model, run_model


# # Global Constants

# These are all the constant variables that we use throughout this notebook. It also specifies the different model variations that were trained. You can select the necessary model at the end of this notebook while running inference.




''' Set the following variable to the path of your dataset (parent directory of actual images) '''
DATA_DIR = '/<path to parent>/'
DATASET_NAME = 'Set14' # Tested on Set5, Set14 and BSDS100

# Directory to store downloaded checkpoints
CHECKPOINT_DIR = './checkpoints/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    
# use_cuda = True  # Whether to use CUDA or CPU
use_cuda = False  # Whether to use CUDA or CPU





MODEL_ARGS = {
    'ABPNRelease': {
        'abpn_28_2x': {
            'num_channels': 28,
            'scaling_factor': 2,
        },
        'abpn_28_3x': {
            'num_channels': 28,
            'scaling_factor': 3,
        },
        'abpn_28_4x': {
            'num_channels': 28,
            'scaling_factor': 4,
        },
        'abpn_32_2x': {
            'num_channels': 32,
            'scaling_factor': 2,
        },
        'abpn_32_3x': {
            'num_channels': 32,
            'scaling_factor': 3,
        },
        'abpn_32_4x': {
            'num_channels': 32,
            'scaling_factor': 4,
        }
    },
    'XLSRRelease': {
        'xlsr_2x': {
            'scaling_factor': 2,
        },
        'xlsr_3x': {
            'scaling_factor': 3,
        },
        'xlsr_4x': {
            'scaling_factor': 4,
        }
    },
    'SESRRelease_M3': {
        'sesr_m3_2x': {
            'scaling_factor': 2
        },
        'sesr_m3_3x': {
            'scaling_factor': 3
        },
        'sesr_m3_4x': {
            'scaling_factor': 4
        },
    },
    'SESRRelease_M5': {
        'sesr_m5_2x': {
            'scaling_factor': 2
        },
        'sesr_m5_3x': {
            'scaling_factor': 3
        },
        'sesr_m5_4x': {
            'scaling_factor': 4
        }
    },
    'SESRRelease_M7': {
        'sesr_m7_2x': {
            'scaling_factor': 2
        },
        'sesr_m7_3x': {
            'scaling_factor': 3
        },
        'sesr_m7_4x': {
            'scaling_factor': 4
        }
    },
    'SESRRelease_M11': {
        'sesr_m11_2x': {
            'scaling_factor': 2
        },
        'sesr_m11_3x': {
            'scaling_factor': 3
        },
        'sesr_m11_4x': {
            'scaling_factor': 4
        }
    },
    'SESRRelease_XL': {
         'sesr_xl_2x': {
            'scaling_factor': 2
        },
        'sesr_xl_3x': {
            'scaling_factor': 3
        },
        'sesr_xl_4x': {
            'scaling_factor': 4
        }
    },
    'QuickSRNetSmall': {
        'quicksrnet_small_1.5x': {
            'scaling_factor': 1.5
        },
        'quicksrnet_small_2x': {
            'scaling_factor': 2
        },
        'quicksrnet_small_3x': {
            'scaling_factor': 3
        },
        'quicksrnet_small_4x': {
            'scaling_factor': 4
        }
    },
    'QuickSRNetMedium': {
        'quicksrnet_medium_1.5x': {
            'scaling_factor': 1.5
        },
        'quicksrnet_medium_2x': {
            'scaling_factor': 2
        },
        'quicksrnet_medium_3x': {
            'scaling_factor': 3
        },
        'quicksrnet_medium_4x': {
            'scaling_factor': 4
        }
    },
    'QuickSRNetLarge': {
        'quicksrnet_large_1.5x': {
            'scaling_factor': 1.5
        },
        'quicksrnet_large_2x': {
            'scaling_factor': 2
        },
        'quicksrnet_large_3x': {
            'scaling_factor': 3
        },
        'quicksrnet_large_4x': {
            'scaling_factor': 4
        }
    }
}

MODELS = list(MODEL_ARGS.keys())


# # Select a model




MODEL_DICT = {}
for idx in range(len(MODELS)):
    MODEL_DICT[idx] = MODELS[idx]

MODEL_DICT


# Select one of the models printed above by selecting the corresponding index in the cell below.




''' Set this variable'''
model_index = 6  # Model index





MODEL_NAME = MODELS[model_index]  # Selected model type

MODEL_SPECS = list(MODEL_ARGS.get(MODEL_NAME).keys())

MODEL_SPECS_DICT = {}
for idx in range(len(MODEL_SPECS)):
    MODEL_SPECS_DICT[idx] = MODEL_SPECS[idx]

MODEL_SPECS_DICT


# Select one of the models printed above by selecting the corresponding index in the cell below




''' Set this variable'''
model_spec_index = 0 # Model specification index





# Choose model
MODEL_CONFIG = MODEL_SPECS[model_spec_index]
print(f'{MODEL_CONFIG} will be used')


# Automatically download model weights:




# Define paths to download filenames
# Do not change the file names
FILENAME_FP32 = 'checkpoint_float32.pth.tar'  # full precision model
FILENAME_INT8 = 'checkpoint_int8.pth' # quantized model
ENCODINGS = 'checkpoint_int8.encodings'
if MODEL_CONFIG.startswith('quicksrnet'):
    ENCODINGS = 'checkpoint_int8_params_only.encodings'  # encodings of the quantized models

# Path to desired model weights and encodings (if necessary)
ENCODING_PATH = os.path.join(CHECKPOINT_DIR, f'release_{MODEL_CONFIG}', ENCODINGS)

# Path to model checkpoint and encodings (if necessary)
MODEL_PATH_INT8 = os.path.join(CHECKPOINT_DIR, f'release_{MODEL_CONFIG}', FILENAME_INT8)
MODEL_PATH_FP32 = os.path.join(CHECKPOINT_DIR, f'release_{MODEL_CONFIG}', FILENAME_FP32)

# AIMET Config
QUANTSIM_CONFIG_FILENAME = 'default_config_per_channel.json'
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, 'default_config_per_channel.json')





if not os.path.exists(MODEL_PATH_INT8) or not os.path.exists(MODEL_PATH_FP32) or not os.path.exists(ENCODING_PATH):
    print('Downloading model weights')
    tar_path = get_tar_path(model_index, model_spec_index) # path to model weights .tar
    urllib.request.urlretrieve(tar_path, 
                               CHECKPOINT_DIR+tar_path.split('/')[-1])
    with tarfile.open(CHECKPOINT_DIR+tar_path.split('/')[-1]) as pth_weights:
          pth_weights.extractall(CHECKPOINT_DIR)
            
if not os.path.exists(CONFIG_PATH):
    QUANTSIM_CONFIG_URL = 'https://raw.githubusercontent.com/quic/aimet/1.23.0/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json'
    urllib.request.urlretrieve(QUANTSIM_CONFIG_URL, "default_config_per_channel.json")


# ## Data loading

# Load test-set images (low-res and high-res pairs)




# Path to test images
TEST_IMAGES_DIR = os.path.join(DATA_DIR, DATASET_NAME)

# Get test images
IMAGES_LR, IMAGES_HR = load_dataset(TEST_IMAGES_DIR, MODEL_ARGS[MODEL_NAME].get(MODEL_CONFIG)['scaling_factor'])


# ## Create model instance and load weights




# Load the model
model_original_fp32 = load_model(MODEL_PATH_FP32, MODEL_NAME, MODEL_ARGS[MODEL_NAME].get(MODEL_CONFIG), 
                   use_quant_sim_model=False, encoding_path=ENCODING_PATH, quantsim_config_path=CONFIG_PATH,
                   calibration_data=IMAGES_LR, use_cuda=use_cuda)

model_original_int8 = load_model(MODEL_PATH_FP32, MODEL_NAME, MODEL_ARGS[MODEL_NAME].get(MODEL_CONFIG), 
                   use_quant_sim_model=True, encoding_path=ENCODING_PATH, quantsim_config_path=CONFIG_PATH,
                   calibration_data=IMAGES_LR, use_cuda=use_cuda)

model_optimized_fp32 = load_model(MODEL_PATH_INT8, MODEL_NAME, MODEL_ARGS[MODEL_NAME].get(MODEL_CONFIG), 
                   use_quant_sim_model=False, encoding_path=ENCODING_PATH, quantsim_config_path=CONFIG_PATH,
                   calibration_data=IMAGES_LR, use_cuda=use_cuda, before_quantization=True)

model_optimized_int8 = load_model(MODEL_PATH_INT8, MODEL_NAME, MODEL_ARGS[MODEL_NAME].get(MODEL_CONFIG), 
                   use_quant_sim_model=True, encoding_path=ENCODING_PATH, quantsim_config_path=CONFIG_PATH,
                   calibration_data=IMAGES_LR, use_cuda=use_cuda, before_quantization=True)


# # Model Inference

# Run inference to get the respective super-resolved images




# Run model inference on test images and get super-resolved images
IMAGES_SR_original_fp32 = run_model(model_original_fp32, IMAGES_LR, use_cuda)
IMAGES_SR_original_int8 = run_model(model_original_int8.model, IMAGES_LR, use_cuda)
IMAGES_SR_optimized_fp32 = run_model(model_optimized_fp32, IMAGES_LR, use_cuda)
IMAGES_SR_optimized_int8 = run_model(model_optimized_int8.model, IMAGES_LR, use_cuda)


# Calculate average-PSNR between the test-set high-res and super-resolved images




# Get the average PSNR for all test-images
avg_psnr = evaluate_average_psnr(IMAGES_SR_original_fp32, IMAGES_HR)
print(f'Original Model | FP32 Environment | Avg. PSNR: {avg_psnr:.3f}')
avg_psnr = evaluate_average_psnr(IMAGES_SR_original_int8, IMAGES_HR)
print(f'Original Model | INT8 Environment | Avg. PSNR: {avg_psnr:.3f}')
avg_psnr = evaluate_average_psnr(IMAGES_SR_optimized_fp32, IMAGES_HR)
print(f'Optimized Model | FP32 Environment | Avg. PSNR: {avg_psnr:.3f}')
avg_psnr = evaluate_average_psnr(IMAGES_SR_optimized_int8, IMAGES_HR)
print(f'Optimized Model | INT8 Environment | Avg. PSNR: {avg_psnr:.3f}')

