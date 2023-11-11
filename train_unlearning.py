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

    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    # Training params
    parser.add_argument('--lr',type = float ,default=5e-4)
    parser.add_argument('--batch_size',type = int ,default=128)
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--device', type=str,help = 'gpu training', default="cuda:0")
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--dataset', type=str,help = 'name of the dataset', default="")

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
    parser.add_argument('--keep_digits', action='store_true', help = 'whether to keep other digits in the remaining dataset')
    parser.add_argument('--weight_reg', action='store_true', help = 'whether to use weight as regularization.')

    args = parser.parse_args()

    return args

def load_checkpoint(directory):

    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    sorted_files = sorted(all_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    max_step = int(sorted_files[-1].split("_")[1].split(".")[0])
    checkpoint_path = os.path.join(directory, f"steps_{max_step:0>8}.pt")
    ckpt = torch.load(checkpoint_path)

    return ckpt


def main(args):

    device = args.device

    if args.dataset == "cifar":

        image_size=32
        base_dim = 128
        channel_mult = [1,2,3,4]
        in_channels = 3
        out_channels = 3

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    elif args.dataset == "mnist":

        image_size=28
        base_dim = 64
        channel_mult = [1,2,4]
        in_channels = 1
        out_channels = 1

        mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
        std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)

    else:
        raise ValueError(f"Unknown dataset {args.dataset}, choose 'cifar' or 'mnist'.")

    model_frozen=DDPM(
        timesteps=args.timesteps,
        base_dim=base_dim,
        channel_mult=channel_mult,
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    ## loss params

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    #torchvision ema setting - https://github.com/pytorch/vision/blob/main/references/classification/train.py#

    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)

    #load checkpoint - old_ckpt=torch.load("/projects/leelab2/mingyulu/unlearning/results/full/models/steps_00042300.pt")

    ckpt = load_checkpoint(f"/data2/mingyulu/data_att/results/{args.dataset}/retrain/models/full/")

    path = f"/data2/mingyulu/data_att/results/{args.dataset}/unlearn_remaining_ablated/"
    params = f"/{excluded_class}/epochs={args.epochs}_datasets={args.keep_digits}_lr={args.lr}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}_weight_reg={args.weight_reg}"

    model_frozen.load_state_dict(ckpt["model"])
    model_frozen.eval()

    freezed_model_dict = ckpt["model"]

    ## Make sure parameters of frozen mdoel is freezed.
    for params in model_frozen.parameters():
        params.requires_grad=False

    for excluded_class in range(10):

        train_dataloader, ablated_dataloader = create_unlearning_dataloaders(
            batch_size=args.batch_size,
            image_size=image_size,
            keep_digits_in_ablated=False,
            keep_digits_in_remaining=False,
            exclude_label=excluded_class
        )

        ## Init new model for unlearning.

        model=DDPM(
            timesteps=args.timesteps,
            base_dim=base_dim,
            channel_mult=channel_mult,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels
        ).to(device)

        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        ckpt = load_checkpoint(f"/data2/mingyulu/data_att/results/{args.dataset}/retrain/models/full/")

        model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])

        optimizer=AdamW(model.parameters(),lr=args.lr)
        scheduler=OneCycleLR(
            optimizer,
            args.lr,
            total_steps=args.epochs*len(train_dataloader),
            pct_start=0.25,
            anneal_strategy='cos'
        )
        loss_fn=nn.MSELoss(reduction='mean')


        global_steps=0

        for epoch in range(args.epochs):

            model.train()

            ## iterating thorugh the size of unlearn dataset.

            for j, ((image_r, _), (image_e, _)) in enumerate(zip(train_dataloader, ablated_dataloader)):

                image_r=image_r.to(device)
                image_e=image_e.to(device)

                ## Sample random noise e_t

                noise=torch.randn_like(image_r).to(device)

                t=torch.randint(0, args.timesteps,(noise.shape[0],)).to(device)

                # get scores for D_r and D_e from the frozen model

                with torch.no_grad():
                    eps_r_frozen = model_frozen(image_r, noise, t)
                    eps_e_frozen = model_frozen(image_e, noise, t)

                # Scores from the fine-tunning model

                eps_r = model(image_r, noise, t)


                # delta logP(D_r) - delta logP(D_e)
                if args.loss_type == "type1":

                    loss = loss_fn(eps_r, alpha1*eps_r_frozen - alpha2*eps_e_frozen)

                elif args.loss_type  == "type2":

                    loss = alpha1*loss_fn(eps_r, eps_r_frozen) - alpha2*loss_fn(eps_r, eps_e_frozen)

                ## weight regularization lambda*(\hat_{\theta} - \theta)
                ## TODO fisher information

                if args.weight_reg:
                    for n, p in model.named_parameters():
                        _loss = (p - freezed_model_dict[n].to(device)) ** 2
                        loss += 0.5 * _loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                ## Update learning rate

                if global_steps%args.model_ema_steps==0:
                    model_ema.update_parameters(model)
                global_steps+=1

                if j % args.log_freq == 0:
                    print(f"Epoch[{epoch+1}/{args.epochs}],Step[{j}/{len(train_dataloader)}],loss:{loss.detach().cpu().item():.5f},lr:{scheduler.get_last_lr()[0]:.6f}")

            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict()
            }

            if (epoch+1)%10 ==0 or (epoch+1)% args.epochs==0:

                model_ema.eval()
                
                os.makedirs(path, exist_ok=True)
                os.makedirs(path + "samples" + params, exist_ok=True)

                samples = model_ema.module.sampling(args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
                save_image(samples, path + "samples" + params + f"/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))

        os.makedirs(path + "models" + params, exist_ok=True)
        torch.save(ckpt,path + "models" + params +  f"/steps_{global_steps:0>8}.pt")

if __name__=="__main__":
    args=parse_args()
    main(args)