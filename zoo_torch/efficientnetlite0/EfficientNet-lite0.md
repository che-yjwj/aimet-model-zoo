# PyTorch EfficientNet-lite0

## Setup AI Model Efficiency Toolkit
Please [install and setup AIMET](https://github.com/quic/aimet/blob/release-aimet-1.21/packaging/install.md) before proceeding further. This model was tested with the `torch_gpu` variant of AIMET version 1.21.0.

## Experiment Setup
### Additional dependencies
Install geffnet using pip install
```
python -m pip install geffnet
```
### Loading AIMET model zoo libraries 
`export PYTHONPATH=$PYTHONPATH:<aimet_model_zoo_path>`

## Model checkpoints and configuration
- Downloading checkpoints and Quantization configuration file are handled through evaluation script.
- The original EfficientNet-lite0 checkpoint can be downloaded from here:
  - https://github.com/rwightman/gen-efficientnet-pytorch
- Optimized EfficientNet-lite0 checkpoint can be downloaded from the [Releases](/../../releases) page.
- The Quantization Simulation (*Quantsim*) Configuration file can be downloaded from here: [default_config_per_channel.json](https://github.com/quic/aimet/blob/17bcc525d6188f177837bbb789ccf55a81f6a1b5/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config_per_channel.json) (Please see [this page](https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/quantization_configuration.html) for more information on this file).

## Dataset
- This evaluation was designed for the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012), which can be obtained from: http://www.image-net.org/
The dataset directory is expected to have 3 subdirectories: train, valid, and test (only the valid test is used, hence if the other subdirectories are missing that is ok).
Each of the {train, valid, test} directories is then expected to have 1000 subdirectories, each containing the images from the 1000 classes present in the ILSVRC2012 dataset, such as in the example below:

```
  train/
  ├── n01440764
  │   ├── n01440764_10026.JPEG
  │   ├── n01440764_10027.JPEG
  │   ├── ......
  ├── ......
  val/
  ├── n01440764
  │   ├── ILSVRC2012_val_00000293.JPEG
  │   ├── ILSVRC2012_val_00002138.JPEG
  │   ├── ......
  ├── ......
```

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
 python3  efficientnetlite0_quanteval.py \
                --default-param-bw <weight bitwidth for quantization - 8 for INT8, 4 for INT4> \
                --dataset-path < path to validation dataset> \
                --batch-size  <batch size as an integer value> \
                --use-cuda <use GPU or CPU> 

```

## Quantization Configuration
- Weight quantization: 8 or 4 bits per channel symmetric quantization
- Bias parameters are not quantized
- Activation quantization: 8 bits, asymmetric quantization
- Model inputs are quantized
- TF_enhanced was used for weight quantization scheme
- TF was used for activation quantization scheme
- Batch norm folding and Adaround have been applied on optimized efficientnet-lite checkpoint
- [Conv - Relu6] layers has been fused as one operation via manual configurations
- 4K Images from ImageNet training dataset (4 images per class) are used as calibration dataset
- Standard ImageNet validation dataset are usef as evaluation dataset
