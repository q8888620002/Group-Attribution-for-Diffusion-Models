class DDPMConfig:

    # CIFAR specific configurations

    cifar_config = {
        "image_size": 32,
        "timesteps": 1000, 
        "dataset": "cifar",
        "base_dim": 128,
        "channel_mult": [1, 2, 3, 4],
        "in_channels": 3,
        "out_channels": 3,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],

        ### Training params

        "lr": 1e-4,
        "batch_size": 80,
        "epochs": 200,
        "model_ema_steps": 10,
        "model_ema_decay" : 0.995
    }

    # MNIST specific configurations

    mnist_config = {
        "image_size": 28,
        "timesteps": 1000, 
        "dataset": "mnist",
        "base_dim": 64,
        "channel_mult": [1, 2, 4],
        "in_channels": 1,
        "out_channels": 1,
        "mean": [0.5],
        "std": [0.5],
        
        ### Training params

        "lr": 1e-4,
        "batch_size": 64,
        "epochs": 100,
        "model_ema_steps": 10,
        "model_ema_decay" : 0.995
    }
