"""Class for TRAK score calculation."""

import torch
import torch.nn as nn

from torch.func import functional_call, vmap, grad
from trak.modelout_functions import AbstractModelOutput
from trak import TRAKer
from pytorch_fid import fid_score, inception


class DTRAKOutput(AbstractModelOutput):
    """
    This a customized DTRAK class for diffusion models.
    """
    def get_output(
        self,
        model: nn.modules,
        samples: torch.Tensor,
        num_train_timesteps: torch.Tensor
    )-> torch.Tensor:
        
        """
        Customized model output, L_simple, for D-TRAK. 

        Args:
            model (torch.nn.Module):
                model
            batch (Iterable[Tensor]):
                input batch
        Returns:
            Tensor:
                model behavior
        """

        noise = torch.randn_like(samples)
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (len(samples),),
        ).long()

        loss_fn = nn.MSELoss(reduction="mean")

        noisy_images = functional_call(model, noise, samples)
        eps = model(noisy_images, timesteps).sample
        loss = loss_fn(eps, noise)

        return loss

    def get_out_to_loss_grad(
        self,
        model: nn.modules,
        samples: torch.Tensor
    ):
        """
        This return an identity matrix.

        Args:
            model (torch.nn.Module): model
            batch (Iterable[Tensor]): input batch

        Returns:
            Identity matrix.

        """
        
        return torch.eyes(len(samples))
    
if __name__ == "__main__":

    traker = TRAKer(
        model=model,
        task=DTRAKOutput,
        train_set_size=TRAIN_SET_SIZE,
        save_dir=SAVE_DIR,
        device=DEVICE,
        proj_dim=1024,
    )

