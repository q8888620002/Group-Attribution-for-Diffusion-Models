"""Configuration for DDPM."""

import os

from src.constants import OUTDIR


class DDPMConfig:
    """DDPM configurations."""

    # CIFAR specific configurations

    cifar_config = {
        "dataset": "cifar",
        "image_size": 32,
        # Training params
        "optimizer_config": {
            "class_name": "Adam",
            "kwargs": {"lr": 1e-4},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 128,
        "training_steps": {
            "retrain": 200000,
            "prune_fine_tune": 200000,
            "ga": 2000,
            "gd": 4000,
            "esd": 5000,
        },
        "ckpt_freq": {
            "retrain": 10000,
            "prune_fine_tune": 10000,
            "ga": 400,
            "gd": 400,
            "esd": 1000,
        },
        "sample_freq": {
            "retrain": 200000,
            "prune_fine_tune": 200000,
            "ga": 2000,
            "gd": 4000,
            "esd": 5000,
        },
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
    cifar2_config = {
        "dataset": "cifar",
        "image_size": 32,
        # Training params
        "optimizer_config": {
            "class_name": "Adam",
            "kwargs": {"lr": 1e-4},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 128,
        "training_steps": {
            "retrain": 20000,
            "prune_fine_tune": 10000,
            "ga": 2000,
            "gd": 4000,
            "esd": 5000,
            "if": 1,
        },
        "ckpt_freq": {
            "retrain": 10000,
            "prune_fine_tune": 10000,
            "ga": 400,
            "gd": 400,
            "esd": 1000,
            "if": 1,
        },
        "sample_freq": {
            "retrain": 2000,
            "prune_fine_tune": 2000,
            "ga": 400,
            "gd": 400,
            "esd": 100,
            "if": 20,
        },
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
    cifar100_config = {
        "dataset": "cifar100",
        "image_size": 32,
        # Training params
        "optimizer_config": {
            "class_name": "Adam",
            "kwargs": {"lr": 1e-4},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 128,
        "training_steps": {
            "retrain": 20000,
            "prune_fine_tune": 10000,
            "ga": 40,
            "gd": 1000,
            "gd_u": 1000,
            "esd": 5000,
            "iu": 1,
        },
        "ckpt_freq": {
            "retrain": 400,
            "prune_fine_tune": 5000,
            "ga": 400,
            "gd": 500,
            "gd_u": 500,
            "esd": 1000,
            "iu": 1,
        },
        "sample_freq": {
            "retrain": 2000,
            "prune_fine_tune": 2000,
            "ga": 400,
            "gd": 500,
            "gd_u": 4000,
            "esd": 100,
            "iu": 20,
        },
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

    cifar100_f_config = {
        "dataset": "cifar100_f",
        "image_size": 32,
        # Training params
        "optimizer_config": {
            "class_name": "Adam",
            "kwargs": {"lr": 1e-4},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 128,
        "training_steps": {
            "retrain": 20000,
            "prune_fine_tune": 20000,
            "ga": 40,
            "gd": 4000,
            "esd": 5000,
            "iu": 1,
        },
        "ckpt_freq": {
            "retrain": 10000,
            "prune_fine_tune": 5000,
            "ga": 400,
            "gd": 500,
            "esd": 1000,
            "iu": 1,
        },
        "sample_freq": {
            "retrain": 2000,
            "prune_fine_tune": 2000,
            "ga": 400,
            "gd": 500,
            "esd": 100,
            "iu": 20,
        },
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
        "optimizer_config": {
            "class_name": "AdamW",
            "kwargs": {"lr": 1.0e-4, "weight_decay": 0.0},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 32,
        "training_steps": {
            "retrain": 20000,
            "prune_fine_tune": 20000,
            "ga": 5,
            "gd": 500,
            "gd_u": 500,
            "esd": 500,
        },
        "ckpt_freq": {
            "retrain": 5000,
            "prune_fine_tune": 5000,
            "ga": 1,
            "gd": 500,
            "gd_u": 500,
            "esd": 100,
        },
        "sample_freq": {
            "retrain": 200000,
            "prune_fine_tune": 200000,
            "ga": 1,
            "gd": 40000,
            "gd_u": 5000,
            "esd": 100,
        },
        "n_samples": 4,
        "unet_config": {
            "_class_name": "UNet2DModel",
            "_diffusers_version": "0.0.4",
            "act_fn": "silu",
            "attention_head_dim": 32,
            "block_out_channels": [224, 448, 672, 896],
            "center_input_sample": False,
            "down_block_types": [
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
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
                "UpBlock2D",
            ],
        },
        "scheduler_config": {
            "_class_name": "DDIMScheduler",
            "_diffusers_version": "0.0.4",
            "beta_end": 0.0195,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.0015,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "trained_betas": None,
        },
        "vqvae_config": {
            "_class_name": "VQModel",
            "_diffusers_version": "0.1.2",
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
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
                "UpDecoderBlock2D",
            ],
        },
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
        "optimizer_config": {
            "class_name": "Adam",
            "kwargs": {"lr": 1e-3, "weight_decay": 0.0},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 64,
        "training_steps": {"retrain": 100, "ga": 5, "gd": 10, "esd": 100},
        "ckpt_freq": {"retrain": 2, "ga": 1, "gd": 1, "esd": 20},
        "sample_freq": {"retrain": 20, "ga": 1, "gd": 1, "esd": 20},
        "n_samples": 500,
    }

    imagenette_config = {
        "dataset": "imagenette",
        "image_size": 256,
        # UNet parameters.
        "unet_config": {
            "_class_name": "UNet2DConditionModel",
            "_diffusers_version": "0.0.4",
            "act_fn": "silu",
            "attention_head_dim": 8,
            "block_out_channels": [320, 640, 1280, 1280],
            "center_input_sample": False,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "out_channels": 4,
            "sample_size": 32,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ],
        },
        # Noise scheduler.
        "scheduler_config": {
            "_class_name": "DDIMScheduler",
            "_diffusers_version": "0.0.4",
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "timestep_values": None,
            "trained_betas": None,
        },
        # Training params.
        "optimizer_config": {
            "class_name": "AdamW",
            "kwargs": {"lr": 1.0e-4, "weight_decay": 1e-6},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "batch_size": 64,  # Largest batch size with fp16 that fits on A40.
        "training_steps": {"retrain": 50000, "ga": 5, "gd": 10, "esd": 150},
        "ckpt_freq": {"retrain": 2500, "ga": 1, "gd": 1, "esd": 50},
        "sample_freq": {"retrain": 2500, "ga": 1, "gd": 1, "esd": 50},
        "n_samples": 60,
    }


class PromptConfig:
    """Prompts for text-to-image generation."""

    artbench_config = {
        "art_nouveau": "an Art Nouveau painting",
        "baroque": "a Baroque painting",
        "expressionism": "an Expressionist painting",
        "impressionism": "an Impressionist painting",
        "post_impressionism": "a Post-Impressionist painting",
        "realism": "a Realist painting",
        "renaissance": "a painting from the Renaissance",
        "romanticism": "a Romanticist painting",
        "surrealism": "a Surrealist painting",
        "ukiyo_e": "a ukiyo-e print",
    }


class LoraTrainingConfig:
    """Training configurations for text_to_image/train_text_to_image_lora.py"""

    artbench_post_impressionism_config = {
        "pretrained_model_name_or_path": "lambdalabs/miniSD-diffusers",
        "resolution": 256,
        "train_batch_size": 64,
        "dataloader_num_workers": 4,
        "checkpointing_steps": 500,
        "resume_from_checkpoint": "latest",
        "checkpoints_total_limit": 1,
        "center_crop": True,
        "random_flip": True,
        "num_train_epochs": 200,
        "learning_rate": 3e-4,
        "lr_scheduler": "cosine",
        "adam_weight_decay": 1e-6,
        "rank": 256,
        "cls_key": "style",
        "cls": "post_impressionism",
    }


class LoraUnlearningConfig:
    """Unlearning configurations for text_to_image/train_text_to_image_lora.py"""

    artbench_post_impressionism_config = {
        "lora_dir": os.path.join(
            OUTDIR, "seed42", "artbench_post_impressionism", "retrain", "models", "full"
        ),
        "max_train_steps": 200,
    }


class LoraSparseUnlearningConfig:
    """Sparse unlearning configurations for text_to_image/train_text_to_image_lora.py"""

    # Config based on the highest SSIM compared to images genearted from the same seeds
    # using the original fully trained model.
    artbench_post_impressionism_config = {
        "lora_dir": os.path.join(
            OUTDIR,
            "seed42",
            "artbench_post_impressionism",
            "pruned_ft_ratio=0.5_lr=3e-05",
            "models",
            "full",
        ),
        "lora_steps": 1580,
        "max_train_steps": 200,
    }


class TextToImageGenerationConfig:
    """Configurations for text_to_image/generate_samples.py"""

    artbench_post_impressionism_config = {
        "pretrained_model_name_or_path": "lambdalabs/miniSD-diffusers",
        "resolution": 256,
        "dataset": "artbench",
        "cls": "post_impressionism",
    }


class TextToImageModelBehaviorConfig:
    """Configurations for text_to_image/compute_model_behaviors.py"""

    artbench_post_impressionism_config = {
        "pretrained_model_name_or_path": "lambdalabs/miniSD-diffusers",
        "dataset": "artbench",
        "cls": "post_impressionism",
        "no_duplicate": True,
        "reference_lora_dir": os.path.join(
            OUTDIR, "seed42", "artbench_post_impressionism", "retrain", "models", "full"
        ),
    }


class DatasetStats:
    """Basic statistics for different datasets."""

    artbench_post_impressionism_stats = {"num_groups": 258}
