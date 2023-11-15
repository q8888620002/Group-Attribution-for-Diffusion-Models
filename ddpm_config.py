class DDPMConfig:

    # CIFAR specific configurations

    # Channel_mult configs from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
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
        "attn_layer":[2],

        ### Training params
        
        ## old -  results/cifar/retrain/models/full/steps_00078200.pt

        # "trained_model": "/projects/leelab/mingyulu/data_att/results/cifar/retrain/models/full/steps_00125000.pt",

        "lr": 1e-4,
        "batch_size": 80,
        "epochs": 200,
        "model_ema_steps": 10,
        "model_ema_decay" : 0.995
    }

    # MNIST specific configurations

    mnist_config = {

        "dataset": "mnist",
        "image_size": 28,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],

        ## Unet params

        "timesteps": 1000, 
        "base_dim": 64,
        "channel_mult": [2, 4],
        "in_channels": 1,
        "out_channels": 1,
        "attn": False,
        "attn_layer":[2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "trained_model": "/projects/leelab2/mingyulu/unlearning/results/full/models/steps_00042300.pt",

        ### Training params

        "lr": 1e-4,
        "batch_size": 64,
        "epochs": 100,
        "model_ema_steps": 10,
        "model_ema_decay" : 0.995
    }
