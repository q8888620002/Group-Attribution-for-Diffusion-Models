"""Configuration for DDPM."""


class DDPMConfig:
    """DDPM configurations."""

    # CIFAR specific configurations

    cifar_config = {
        "dataset": "cifar",
        "image_size": 32,
        "trained_model": (
            "/projects/leelab/mingyulu/data_att/results/cifar/"
            "retrain/models/full/steps_00125000.pt",
        ),
        # Training params
        "lr": 1e-4,
        "batch_size": 128,
        "epochs": {"retrain": 800, "ga": 5, "gd": 10, "esd": 500},
        "ckpt_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 250},
        "sample_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 20},
        "n_samples": 64,
        "unet_config": {
            "_class_name": "UNet2DModel",
            "_diffusers_version": "0.24.0",
            "act_fn": "silu",
            "add_attention": True,
            "attention_head_dim": None,
            "attn_norm_num_groups": None,
            "block_out_channels": [128, 256, 256, 256],
            "center_input_sample": False,
            "class_embed_type": None,
            "down_block_types": [
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
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
            "up_block_types": ["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "upsample_type": "conv",
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
            "variance_type": "fixed_large",
        },
    }

    # CelebA-HQ specific configurations

    celeba_config = {
        "dataset": "celeba",
        "image_size": 256,
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": {"retrain": 800, "ga": 5, "gd": 10, "esd": 500},
        "ckpt_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 100},
        "sample_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 100},
        "n_samples": 32,
        "unet_config": {
            "_class_name": "UNet2DModel",
            "_diffusers_version": "0.0.4",
            "act_fn": "silu",
            "attention_head_dim": 32,
            "block_out_channels": [
                224,
                448,
                672,
                896
            ],
            "center_input_sample": False,
            "down_block_types": [
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 3,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 64,
            "time_embedding_type": "positional",
            "up_block_types": [
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ]
        },
        "scheduler_config":{
            "_class_name": "DDIMScheduler",
            "_diffusers_version": "0.0.4",
            "beta_end": 0.0195,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.0015,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "trained_betas": None
            },

        "vqvae_config": {
            "_class_name": "VQModel",
            "_diffusers_version": "0.1.2",
            "act_fn": "silu",
            "block_out_channels": [
                128,
                256,
                512
            ],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            "in_channels": 3,
            "latent_channels": 3,
            "layers_per_block": 2,
            "num_vq_embeddings": 8192,
            "out_channels": 3,
            "sample_size": 256,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ]
            }
    }

    # MNIST specific configurations
    # Reference: https://colab.research.google.com/github/st-howard/blog-notebooks/blob/
    # main/MNIST-Diffusion/Diffusion%20Digits%20-%20Generating%20MNIST%20Digits%20
    # from%20noise%20with%20HuggingFace%20Diffusers.ipynb

    mnist_config = {
        "dataset": "mnist",
        "image_size": 28,
        # UNet parameters.
        "unet_config": {
            "_class_name": "UNet2DModel",
            "_diffusers_version": "0.24.0",
            "sample_size": 32,
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": [128, 128, 256, 512],
            "down_block_types": [
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ],
            "up_block_types": [
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
        },
        "trained_model": (
            "/projects/leelab/mingyulu/data_att/results/mnist/"
            "retrain/models/full/steps_00065660.pt"
        ),
        # Noise scheduler.
        "scheduler_config": {
            "_class_name": "DDPMScheduler",
            "_diffusers_version": "0.24.0",
            "num_train_timesteps": 1000,
        },
        # Training params
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": {"retrain": 100, "ga": 5, "gd": 10, "esd": 100},
        "ckpt_freq": {"retrain": 2, "ga": 1, "gd": 1, "esd": 20},
        "sample_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 20},
        "n_samples": 500,
    }
