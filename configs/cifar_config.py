{
    "_class_name": "UNet2DModel",
    "_diffusers_version": "0.0.4",
    "act_fn": "silu",
    "attention_head_dim": None,
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
  }