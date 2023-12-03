"""Configuration for DDPM."""


class DDPMConfig:
    """Configuration class for DDPM"""

    # CIFAR specific configurations

    # Channel_mult configs from
    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    #     if image_size == 256:
    #       channel_mult = (1, 1, 2, 2, 4, 4)
    #     elif image_size == 64:
    #       channel_mult = (1, 2, 3, 4)
    #     elif image_size == 32:
    #       channel_mult = (1, 2, 2, 2)

    cifar_config = {
        "dataset": "cifar",
        "image_size": 32,
        # "mean": [0.485, 0.456, 0.406],
        # "std": [0.229, 0.224, 0.225],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        # Unet params
        "timesteps": 1000,
        "base_dim": 128,
        "channel_mult": [1, 2, 2, 2],
        "in_channels": 3,
        "out_channels": 3,
        "num_res_blocks": 2,
        "dropout": 0.15,
        "attn": True,
        "attn_layer": [2],
        # Training params

        "trained_model": (
            "/projects/leelab/mingyulu/data_att/results/cifar/"
            "retrain/models/full/steps_00125000.pt"
        ),
        "lr": 1e-4,
        "batch_size": 128,
        "epochs": {"retrain": 250, "ga": 5, "gd": 10, "esd": 250},
        "model_ema_steps": 10,
        "model_ema_decay": 0.995,
        "ckpt_freq": {"retrain": 50, "ga": 1, "gd": 1, "esd": 20},
        "sample_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 20},
        "n_samples": 64,

        # Model configs from https://huggingface.co/google/ddpm-cifar10-32/tree/main
        # Cifar-10 Checkpoints - https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/?p=%2Fdiffusion_models_converted%2Fema_diffusion_cifar10_model&mode=list
        
        "unet_config": {
            "_class_name": "UNet2DModel",
            "_diffusers_version": "0.0.4",
            "act_fn": "silu",
            "block_out_channels": [
              128,
              256,
              256,
              256
            ],
            "center_input_sample": False,
            "down_block_types": [
              "DownBlock2D",
              "AttnDownBlock2D",
              "DownBlock2D",
              "DownBlock2D"
            ],
            "downsample_padding": 0,
            "flip_sin_to_cos": False,
            "freq_shift": 1,
            "in_channels": 3,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-06,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 32,
            "time_embedding_type": "positional",
            "up_block_types": [
              "UpBlock2D",
              "UpBlock2D",
              "AttnUpBlock2D",
              "UpBlock2D"
            ]
        },

        "scheduler_config": {
          "_class_name": "DDPMScheduler",
          "_diffusers_version": "0.1.1",
          "beta_end": 0.02,
          "beta_schedule": "linear",
          "beta_start": 0.0001,
          "clip_sample": True,
          "num_train_timesteps": 1000,
          "trained_betas": None,
          "variance_type": "fixed_large"
        }
    }

    # MNIST specific configurations

    mnist_config = {
        "dataset": "mnist",
        "image_size": 28,
        "mean": [0.5],
        "std": [0.5],
        # Unet params
        "timesteps": 1000,
        "base_dim": 64,
        "channel_mult": [1, 2],
        "in_channels": 1,
        "out_channels": 1,
        "attn": False,
        "attn_layer": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "trained_model": (
            "/projects/leelab/mingyulu/data_att/results/mnist/"
            "retrain/models/full/steps_00065660.pt"
        ),
        # Training params
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": {"retrain": 100, "ga": 5, "gd": 10,"esd": 100},
        "model_ema_steps": 10,
        "model_ema_decay": 0.995,
        "ckpt_freq": {"retrain": 2, "ga": 1, "gd": 1, "esd": 20},
        "sample_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 20},
        "n_samples": 500,
    }
