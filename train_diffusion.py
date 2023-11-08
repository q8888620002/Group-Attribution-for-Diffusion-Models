import os
import math
import argparse

import torch
import torch.nn as nn

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from diffusion.diffusions import DDPM
from utils import *



def parse_args():
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--device', type=str ,help = 'device of training', default="cuda:0")
    parser.add_argument('--dataset',type = str, help="dataset name", default = '')


    args = parser.parse_args()

    return args


def main(args):

    device = args.device

    for excluded_class in range(10, -1, -1):

        if excluded_class == 10:
            excluded_class = None

        if args.dataset == "cifar":

            image_size=32

            model=DDPM(
                timesteps=args.timesteps,
                base_dim=128,
                channel_mult=[1,2,3,4],
                image_size=image_size,
                in_channels=3,
                out_channels=3
            ).to(device)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        elif args.dataset == "mnist":

            image_size=32

            model=DDPM(
                timesteps=args.timesteps,
                base_dim=64,
                channel_mult=[1,2,4,8],
                image_size=image_size,
                in_channels=1,
                out_channels=1
            ).to(device)

            mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
            std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)

        else:
            raise ValueError(f"Unknown dataset {args.dataset}, choose 'cifar' or 'mnist'.")


        train_dataloader, _ = create_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            image_size=image_size,
            excluded_class=excluded_class
        )


        #torchvision ema setting
        #https://github.com/pytorch/vision/blob/main/references/classification/train.py#

        adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        optimizer=AdamW(model.parameters(),lr=args.lr)
        scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
        loss_fn=nn.MSELoss(reduction='mean')

        #load checkpoint

        if args.ckpt:
            ckpt=torch.load(args.ckpt)
            model_ema.load_state_dict(ckpt["model_ema"])
            model.load_state_dict(ckpt["model"])

        global_steps=0

        total_steps = len(train_dataloader) * args.epochs

        fid_scores = []

        for i in range(args.epochs):
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

                if global_steps%args.model_ema_steps==0:
                    model_ema.update_parameters(model)
                global_steps+=1

                if j % args.log_freq == 0:

                    print(f"Epoch[{i+1}/{args.epochs}],Step[{j}/{len(train_dataloader)}],loss:{loss.detach().cpu().item():.5f},lr:{scheduler.get_last_lr()[0]:.5f}")

                if global_steps > 0 and global_steps % (total_steps//4) == 0:

                    ## Checkpoints for training

                    print(f"Checkpoint saved at step {global_steps}")

                    ckpt = {
                        "model": model.state_dict(),
                        "model_ema": model_ema.state_dict()
                    }

                    os.makedirs(f"results/{args.dataset}/retrain/models/{excluded_class}", exist_ok=True)
                    torch.save(ckpt, f"results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.pt")

                ## adding images for FID evaluation

                if real_images.size(0) < args.n_samples:
                    with torch.no_grad():
                        real_images = torch.cat((real_images, image), dim=0)

            if excluded_class is None:
                excluded_class = "full"

            model_ema.eval()

            samples = model_ema.module.sampling(args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)

            ## Calculating Fid Score

            if real_images.size(0) > args.n_samples:
                real_images = real_images[:args.n_samples]

            fid_value = calculate_fid(samples, real_images, device)
            fid_scores.append(fid_value)

            print(f"FID score after {global_steps} steps: {fid_value}")

            samples = samples * std + mean

            os.makedirs(f"results/{args.dataset}/retrain/samples/{excluded_class}", exist_ok=True)
            save_image(samples, f"results/{args.dataset}/retrain/samples/{excluded_class}/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))

        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }

        torch.save(ckpt, f"results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.pt")

        np.save(f"results/{args.dataset}/retrain/models/{excluded_class}/steps_{global_steps:0>8}.npy",  np.array(fid_scores))

if __name__=="__main__":
    args=parse_args()
    main(args)