from .diffusions import DDPM


def create_ddpm_model(config):
    """
    Helper function for DDPM init
    """
    return DDPM(
        timesteps=config["timesteps"],
        base_dim=config["base_dim"],
        channel_mult=config["channel_mult"],
        image_size=config["image_size"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
    )
