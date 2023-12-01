import os
import math
import argparse

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from utils import *
from ddpm_config import DDPMConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils import make_image_grid

unet_config = {
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
  }

scheduler_config = {
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



def parse_args():
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--dataset', type=str, default='' )

    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--device', type=str ,help = 'device of training', default="cuda:1")


    args = parser.parse_args()

    return args



def main(args):

    device = args.device

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset  == "mnist":
        config = {**DDPMConfig.mnist_config}
    else:
        raise ValueError(f"Unknown dataset {config['dataset']}, choose 'cifar' or 'mnist'.")

    for excluded_class in range(10, -1, -1):

        excluded_class = None if excluded_class == 10 else excluded_class

        train_dataloader, _ = create_dataloaders(
            dataset_name=config["dataset"],
            batch_size=config["batch_size"],
            excluded_class=excluded_class,
            unlearning=False
        )

        pipeline = DDPMPipeline(
            unet=UNet2DModel(
                **unet_config
            ).to(device),
            scheduler=DDPMScheduler(**scheduler_config)
        )

        model = pipeline.unet
        noise_scheduler = pipeline.scheduler

        adjust = 1* config['batch_size'] *config['model_ema_steps'] / config['epochs']
        alpha = 1.0 - config['model_ema_decay']
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        optimizer=AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=1e-4
        )

        scheduler=OneCycleLR(
            optimizer,
            config["lr"],
            total_steps=config['epochs']*len(train_dataloader),
            pct_start=0.25,
            anneal_strategy='cos'
        )

        loss_fn=nn.MSELoss(reduction='mean')

        global_steps = 0

        for epoch in range(config['epochs']):

            model.train()

            pipeline = DDIMPipeline(
                unet= model ,
                scheduler=DDIMScheduler(num_train_timesteps=config["timesteps"])
            )

            pipeline.scheduler.set_timesteps(config["timesteps"])

            images = pipeline(
                batch_size=args.n_samples,
                num_inference_steps=config["timesteps"],
            ).images

            image_grid = make_image_grid(
                images,
                rows=int(math.sqrt(args.n_samples)),
                cols=int(math.sqrt(args.n_samples))
            )

            os.makedirs(f"results/{args.dataset}/retrain/samples/{excluded_class}", exist_ok=True)
            image_grid.save(f"results/{args.dataset}/retrain/samples/{excluded_class}/steps_{global_steps:0>8}.png")

            for j, (image, _) in enumerate(train_dataloader):

                noise=torch.randn_like(image).to(device)
                image=image.to(device)

                timesteps = torch.randint(
                   0, config["timesteps"], (image.size()[0],), device=image.device
                ).long()

                timesteps = torch.cat([timesteps, noise_scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:image.size()[0]]

                noisy_images = noise_scheduler.add_noise(image, noise, timesteps)

                optimizer.zero_grad()

                pred = model(noisy_images, timesteps).sample

                loss=loss_fn(pred,noise)
                loss.backward()

                optimizer.step()
                scheduler.step()

                if global_steps%config['model_ema_steps']==0:
                    model_ema.update_parameters(model)

                if j % args.log_freq == 0:
                    print(f"Epoch[{epoch+1}/{config['epochs']}],Step[{j}/{len(train_dataloader)}], loss:{loss.detach().cpu().item():.5f}, lr:{scheduler.get_last_lr()[0]:.6f}")

                global_steps+=1

            ## Generate samples and calculate fid score for non-mnist dataset every 20 epochs

            excluded_class = "full" if excluded_class is None else excluded_class

            if  (epoch+1) % 20 == 0 or global_steps == config['epochs']*len(train_dataloader):
                model_ema.eval()

                print("Sampling images...")

                pipeline = DDPMPipeline(
                    unet= model_ema.module ,
                    scheduler=DDIMScheduler(num_train_timesteps=config["timesteps"])
                )

                pipeline.scheduler.set_timesteps(config["timesteps"])

                images = pipeline(
                    batch_size=args.n_samples,
                    num_inference_steps=config["timesteps"],
                    output_type="numpy",
                ).images

                images = np.clip(images / 2 + 0.5, 0. , 1.).squeeze()

                os.makedirs(f"results/{args.dataset}/retrain/samples/{excluded_class}", exist_ok=True)

                save_image(
                    torch.from_numpy(images).permute([0, 3, 1, 2]),
                    f"results/{args.dataset}/retrain/samples/{excluded_class}/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples))
                )

            ## Checkpoints for training

            if  (epoch+1) % (config['epochs'] //2) == 0 or (epoch+1) % config['epochs'] == 0:

                print(f"Checkpoint saved at step {global_steps}")

                os.makedirs(f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}", exist_ok=True)
                torch.save(model, f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/unet_ema_pruned-{global_steps:0>8}.pt" )

        torch.save(model, f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/unet_ema_pruned-{global_steps:0>8}.pt" )

        if config['dataset'] != "mnist":
            np.save(f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.npy",  np.array(fid_scores))

if __name__=="__main__":
    args=parse_args()
    main(args)