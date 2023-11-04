import os
import math
import argparse

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize, ToTensor, Compose, Lambda, Grayscale
from scipy.linalg import sqrtm


from model import MNISTDiffusion
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
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
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args

def main(args):
    device="cpu" if args.cpu else "cuda"

    model_full=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    model_ablated=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    model_unlearn=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #load checkpoint

    ckpt=torch.load("results/full/models/steps_00042300.pt")
    model_full.load_state_dict(ckpt["model"])
    model_full.eval()

    ckpt=torch.load("results/ablated/2/models/steps_00042300.pt")
    model_ablated.load_state_dict(ckpt["model"])
    model_ablated.eval()

    ckpt=torch.load("results/unlearn/2/models/steps_00005300.pt")
    model_unlearn.load_state_dict(ckpt["model"])
    model_unlearn.eval()

    global_steps=0

    x_t=torch.randn((args.n_samples, 1, 28, 28)).to(device)

    samples_original = model_full._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
    samples_ablated = model_ablated._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
    samples_unlearn = model_unlearn._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)

    os.makedirs(f"results/generated_samples/original/samples", exist_ok=True)
    os.makedirs(f"results/generated_samples/ablated/samples", exist_ok=True)
    os.makedirs(f"results/generated_samples/unlearn/samples", exist_ok=True)

    save_image(samples_original, f"results/generated_samples/original/samples/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))
    save_image(samples_ablated, f"results/generated_samples/ablated/samples/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))
    save_image(samples_unlearn, f"results/generated_samples/unlearn/samples/steps_{global_steps:0>8}.png", nrow=int(math.sqrt(args.n_samples)))

    # print(calculate_fid_mnist(samples_unlearn, samples_original), calculate_fid_mnist(samples_ablated, samples_original), calculate_fid_mnist(samples_ablated, samples_unlearn))

    print(clip_score(samples_unlearn, samples_original), clip_score(samples_ablated, samples_original))



if __name__=="__main__":
    args=parse_args()
    main(args)