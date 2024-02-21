## Converting Pre-trained CIFAR-10 Checkpoint for DDPMpipeline on Huggingface

To convert a pre-trained CIFAR-10 checkpoint for compatibility with the DDPMpipeline on Huggingface, follow these steps:

### Step 1: Prepare the Pretrained Folder
Create a directory for your pretrained model files:
```bash
mkdir -p [your_pretrained_folder]
```
Replace `[your_pretrained_folder]` with your desired folder name.

### Step 2: Download the Weights
Download the pre-trained weights from the following link:
[Download Pre-trained Weights](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/?p=%2Fdiffusion_models_converted%2Fdiffusion_cifar10_model&mode=list).

### Step 3: Convert the Checkpoint
Convert the original DDPM checkpoint to the Diffusers format for CIFAR-10:
```bash
python tools/convert_ddpm_original_checkpoint_to_diffusers_cifar10.py \
    --checkpoint_path [your_pretrained_folder]/cifar10-ema-model-790000.ckpt \
    --config_file tools/ddpm_cifar10_config.json \
    --dump_path [your_pretrained_folder]/ddpm_ema_cifar10
```
Replace `[your_pretrained_folder]` with your folder name from Step 1.
