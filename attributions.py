"""Class for TRAK score calculation."""

import torch
import torch.nn as nn

from torch.func import functional_call, vmap, grad
from trak.modelout_functions import AbstractModelOutput
from pytorch_fid import fid_score, inception


class DiffusionOutput(AbstractModelOutput):
    """
    This a customized TRAK class that output FID score as model behavior.
    """
    def get_output(
        self,
        model,
        samples,
        pipeline_scheduler,
        noise,
        timesteps,
        device
    )-> float:
        noisy_images_r = pipeline_scheduler.add_noise(samples, noise, timesteps)

        output = model(noisy_images_r, timesteps).sample

        return output

    def get_out_to_loss_grad(
        self,
        model,
        samples,
        device,
        pipeline_scheduler
    ):
        """
        This is a customized function for derivative of delta L/F
        """

        noise = torch.randn_like(samples).to(device)
        timesteps = torch.randint(
            0,
            pipeline_scheduler.config.num_train_timesteps,
            (len(samples) // 2 + 1,),
            device=samples.device,
        ).long()
        timesteps = torch.cat(
            [
                timesteps,
                pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
            ],
            dim=0,
        )[: len(samples)]

        loss_fn = nn.MSELoss(reduction="mean")

        noisy_images_r = functional_call(model, noise, samples)

        eps_r = model(noisy_images_r, timesteps).sample
        loss = loss_fn(eps_r, noise)

        return None