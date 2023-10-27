import os
import math
import argparse

import torch
import torch.nn as nn

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from model import MNISTDiffusion
from utils import *



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    # Training params
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--device', type=str,help = 'gpu training', default="cuda:0")
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)

    ## Diffusion params

    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')

    ## Loss related params

    parser.add_argument('--loss_type',type = str,help = 'define loss type',default='type1')
    parser.add_argument('--alpha1',type = float,help = 'loss params: alpha1',default=1)
    parser.add_argument('--alpha2',type = float,help = 'loss params: alpha2',default=1e-1)
    parser.add_argument('--keep_digits', action='store_true', help = 'whether to keep other digits in the ablated dataset')

    args = parser.parse_args()

    return args


def main(args):

    # device="cpu" if args.cpu else "cuda:0"
    device = args.device

    for digit in range(1, 10):

        train_dataloader, ablated_dataloader = create_unlearning_dataloaders_v2(
            batch_size=args.batch_size,
            image_size=28,
            keep_digits=args.keep_digits,
            exclude_label=digit
        )

        model=MNISTDiffusion(
            timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

        model_frozen=MNISTDiffusion(timesteps=args.timesteps,
                    image_size=28,
                    in_channels=1,
                    base_dim=args.model_base_dim,
                    dim_mults=[2,4]).to(device)

        #torchvision ema setting
        #https://github.com/pytorch/vision/blob/main/references/classification/train.py#

        adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        optimizer=AdamW(model.parameters(),lr=args.lr)
        scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
        loss_fn=nn.MSELoss(reduction='mean')

        alpha1 = args.alpha1
        alpha2 = args.alpha2
        #load checkpoint

        ckpt=torch.load("results/full/models/steps_00042300.pt")

        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

        model_frozen.load_state_dict(ckpt["model"])
        model_frozen.eval()

        ## Make sure parameters of frozen mdoel is freezed.

        for params in model_frozen.parameters():
            params.require_grad=False

        global_steps=0

        for i in range(args.epochs):

            model.train()

            ## iterating thorugh the size of unlearn dataset.

            for j, ((image_r, _), (image_e, _)) in enumerate(zip(train_dataloader, ablated_dataloader)):

                image_r=image_r.to(device)
                image_e=image_e.to(device)

                ## Sample random noise e_t

                noise=torch.randn_like(image_r).to(device)
                t=torch.randint(0, args.timesteps,(image_e.shape[0],)).to(device)

                with torch.no_grad():

                    # get scores for D_r and D_e from the frozen model

                    eps_e_frozen = model_frozen(image_e, noise, t)
                    eps_r_frozen = model_frozen(image_r, noise, t)

                # Scores from the fine-tunning model

                eps_r = model(image_r, noise, t)

                if args.loss_type == "type1":
                    # delta logP(D_r) - delta logP(D_e)
                    loss = loss_fn(eps_r, eps_r_frozen - alpha2*eps_e_frozen)

                elif args.loss_type == "type2":
                    # delta logP(D_r) - delta logP(D_e)
                    loss =loss_fn(eps_r, eps_r_frozen - alpha2*eps_e_frozen)

                    # delta logP(D_r)
                    loss2 =loss_fn(eps_r, eps_r_frozen)
                    loss = alpha1*loss + loss2

                elif args.loss_type  == "type3":
                    loss = loss_fn(eps_r, (1-alpha2)*eps_e_frozen)

                elif args.loss_type  == "type4":
                    loss = loss_fn(eps_r, eps_r_frozen)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                ## Update learning rate

                if global_steps%args.model_ema_steps==0:
                    model_ema.update_parameters(model)
                global_steps+=1

                if j % args.log_freq == 0:
                    print(f"Epoch[{i+1}/{args.epochs}],Step[{j}/{len(train_dataloader)}],loss:{loss.detach().cpu().item():.5f},lr:{scheduler.get_last_lr()[0]:.5f}")

            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict()
            }

            path = f"results/unlearn_remaining_ablated/{digit}/epochs={args.epochs}_datasets={args.keep_digits}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}"

            os.makedirs(path, exist_ok=True)
            os.makedirs(path+"/models", exist_ok=True)
            os.makedirs(path+"/samples", exist_ok=True)

            torch.save(ckpt, path+ f"/models/steps_{global_steps:0>8}.pt")

            model_ema.eval()
            samples = model_ema.module.sampling(args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
            save_image(samples, path + f"/samples/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)