"""Configuration for DDPM."""


class DDPMConfig:

    # CIFAR specific configurations

    cifar_config = {
        "dataset": "cifar",
        "image_size": 32,

        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],


        "trained_model": (
            "/projects/leelab/mingyulu/data_att/results/cifar/"
            "retrain/models/full/steps_00125000.pt",
        ),

        # Training params
        "lr": 1e-4,
        "batch_size": 128,
        "epochs": {"retrain": 400, "ga": 5, "gd": 10, "esd": 500},
        "ckpt_freq": {"retrain": 100, "ga": 1, "gd": 1, "esd": 250},
        "sample_freq": {"retrain": 50, "ga": 1, "gd": 1, "esd": 20},
        "n_samples": 64,

        # Cifar-10 Checkpoints - https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/?p=%2Fdiffusion_models_converted%2Fema_diffusion_cifar10_model&mode=list

        "unet_config": {
            "_class_name": "UNet2DModel",
            "_diffusers_version": "0.24.0",
            "act_fn": "silu",
            "add_attention": True,
            "attention_head_dim": None,
            "attn_norm_num_groups": None,
            "block_out_channels": [
              128,
              256,
              256,
              256
            ],
            "center_input_sample": False,
            "class_embed_type": None,                                                                           
            "down_block_types": [
              "DownBlock2D",
              "AttnDownBlock2D",
              "DownBlock2D",
              "DownBlock2D"
            ],
            "downsample_padding": 0,
            "downsample_type": "conv",
            "dropout": 0.0,
            "flip_sin_to_cos": False,
            "freq_shift": 1,
            "in_channels": 3,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-06,
            "norm_num_groups": 32,
            "num_class_embeds": None,
            "num_train_timesteps": None,
            "out_channels": 3,
            "resnet_time_scale_shift": "default",
            "sample_size": 32,
            "time_embedding_type": "positional",
            "up_block_types": [
              "UpBlock2D",
              "UpBlock2D",
              "AttnUpBlock2D",
              "UpBlock2D"
            ],
            "upsample_type": "conv"
          },

        "scheduler_config": {
            "_class_name": "DDPMScheduler",
            "_diffusers_version": "0.24.0",
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "clip_sample": True,
            "clip_sample_range": 1.0,
            "dynamic_thresholding_ratio": 0.995,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "steps_offset": 0,
            "thresholding": False,
            "timestep_spacing": "leading",
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
