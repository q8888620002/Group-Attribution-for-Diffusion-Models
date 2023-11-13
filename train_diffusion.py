import os
import math
import argparse

import torch
import torch.nn as nn

from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from ddpm_config import DDPMConfig
from diffusion.diffusions import DDPM
from utils import *



def parse_args():
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--dataset', type=str, default='' )

    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--device', type=str ,help = 'device of training', default="cuda:0")


    args = parser.parse_args()

    return args


def main(args):

    device = args.device

    for excluded_class in range(10, -1 ,-1):

        if args.dataset == "cifar":
            config = {**DDPMConfig.cifar_config}
        elif args.dataset  == "mnist":
            config = {**DDPMConfig.mnist_config}
        else:
            raise ValueError(f"Unknown dataset {config['dataset']}, choose 'cifar' or 'mnist'.")

        model = DDPM(
            timesteps=config['timesteps'],
            base_dim=config['base_dim'],
            channel_mult=config['channel_mult'],
            image_size=config['image_size'],
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            attn=config['attn'],
            attn_layer=config['attn_layer'],
            num_res_blocks=config['num_res_blocks'],
            dropout=config['dropout']
        ).to(device)

        mean = torch.tensor(config['mean']).view(1, -1, 1, 1).to(device)
        std = torch.tensor(config['std']).view(1, -1, 1, 1).to(device)

        excluded_class = None if excluded_class== 10 else excluded_class

        train_dataloader, _ = create_dataloaders(
            dataset_name=config["dataset"],
            batch_size=config["batch_size"],
            excluded_class=excluded_class,
            unlearning=False
        )

        #torchvision ema setting
        #https://github.com/pytorch/vision/blob/main/references/classification/train.py#

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

        #load checkpoint

        if args.ckpt:
            ckpt=torch.load(args.ckpt)
            model_ema.load_state_dict(ckpt["model_ema"])
            model.load_state_dict(ckpt["model"])

        global_steps=0

        fid_scores = []

        for epoch in range(config['epochs']):

            model.train()
            real_images = torch.Tensor().to(device)

            for j, (image, _) in enumerate(train_dataloader):

                noise=torch.randn_like(image).to(device)
                image=image.to(device)
                pred=model(image,noise)

                loss=loss_fn(pred,noise)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if global_steps%config['model_ema_steps']==0:
                    model_ema.update_parameters(model)

                if j % args.log_freq == 0:
                    print(f"Epoch[{epoch+1}/{config['epochs']}],Step[{j}/{len(train_dataloader)}], loss:{loss.detach().cpu().item():.5f}, lr:{scheduler.get_last_lr()[0]:.6f}")

                ## adding images for FID evaluation

                if config['dataset'] != "mnist":
                    if real_images.size(0) < args.n_samples:
                        with torch.no_grad():
                            real_images = torch.cat((real_images, image), dim=0)

                    if real_images.size(0) > args.n_samples:
                        real_images = real_images[:args.n_samples]

                global_steps+=1

            ## Generate samples and calculate fid score for non-mnist dataset every 15 epochs
            excluded_class = "full" if excluded_class is None else excluded_class

            if  (epoch+1) % 15 == 0 or (epoch+1) % config['epochs'] == 0 or global_steps == config['epochs']*len(train_dataloader):
                
                model_ema.eval()
                samples = model_ema.module.sampling(
                    args.n_samples, 
                    clipped_reverse_diffusion=not args.no_clip, 
                    device=device
                )

                if config["dataset"] != "mnist":
                    fid_value = calculate_fid(samples, real_images, device)
                    fid_scores.append(fid_value)
                    print(f"FID score after {global_steps} steps: {fid_value}")
                
                os.makedirs(f"results/{args.dataset}/retrain/samples/{excluded_class}", exist_ok=True)
                save_image(samples, f"results/{args.dataset}/retrain/samples/{excluded_class}/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))


            ## Checkpoints for training

            if  (epoch+1) % (config['epochs'] //2) == 0 or (epoch+1) % config['epochs'] == 0:

                print(f"Checkpoint saved at step {global_steps}")

                os.makedirs(f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}", exist_ok=True)
                ckpt = {
                    "model": model.state_dict(),
                    "model_ema": model_ema.state_dict()
                }
                torch.save(ckpt, f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.pt")


        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }


        torch.save(ckpt, f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.pt")

        if config['dataset'] != "mnist":
            np.save(f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.npy",  np.array(fid_scores))


if __name__=="__main__":
    args=parse_args()
    main(args)