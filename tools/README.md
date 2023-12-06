# Converting Pre-trained CIFAR-10 Checkpoint for DDPMpipeline on Huggingface

Follow these steps to convert a pre-trained CIFAR-10 checkpoint into a format compatible with the DDPMpipeline from Huggingface:

### Step 1: Prepare the Pretrained Folder
Create a directory for your pretrained model files:
```bash
mkdir -p your_pretrained_folder
```

### Step 2: Download the Weights
Download the pre-trained weights from [this link](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/?p=%2Fdiffusion_models_converted%2Fdiffusion_cifar10_model&mode=list).

### Step 3: Convert the Checkpoint
Use the following command to convert the original DDPM checkpoint to the Diffusers format for CIFAR-10:
```bash
python tools/convert_ddpm_original_checkpoint_to_diffusers_cifar10.py \
    --checkpoint_path pretrained/cifar10-ema-model-790000.ckpt \
    --config_file tools/ddpm_cifar10_config.json \
    --dump_path pretrained/ddpm_ema_cifar10
```
