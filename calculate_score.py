import os
import math
import argparse
import glob

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


from model import DDPM
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--dataset',type = str,help = 'dataset name',default='')

    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--device', type= str , help = 'device to train')

    args = parser.parse_args()

    return args

def main(args):

    dataset_configs = {
        "mnist": {
            "image_size": 28,
            "in_channels": 1,
            "base_dim": 64,
            "dim_mults": [2, 4],
        },
        "cifar": {
            "image_size": 32,
            "in_channels": 3,
            "base_dim": 128,
            "dim_mults": [1, 2, 4, 8],
        },
    }

    config = dataset_configs.get(args.dataset)

    if config is None:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    model_full = DDPM(timesteps=args.timesteps, **config).to(device)
    model_ablated = DDPM(timesteps=args.timesteps, **config).to(device)
    model_unlearn = DDPM(timesteps=args.timesteps, **config).to(device)

    #load checkpoint


    def find_max_step_file(path_pattern):
        files = glob.glob(path_pattern)
        max_step_file = max(files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        return max_step_file

    if args.dataset == "mnist":
        max_step_file = find_max_step_file("/projects/leelab2/mingyulu/unlearning/results/full/models/steps_*.pt")
        ckpt = torch.load(max_step_file)
        model_full.load_state_dict(ckpt["model"])
        model_full.eval()

    for i in range(1, 10):
        max_step_file_ablated = find_max_step_file(f"/projects/leelab2/mingyulu/unlearning/results/ablated/{i}/models/steps_*.pt")
        ckpt_ablated = torch.load(max_step_file_ablated)
        model_ablated.load_state_dict(ckpt_ablated["model"])
        model_ablated.eval()

        max_step_file_unlearn = find_max_step_file(f"results/models/unlearn_remaining_ablated/{i}/epochs=100_datasets=False_loss=type1:alpha1=1.0_alpha2=0.01_weight_reg=False/steps_*.pt")
        ckpt_unlearn = torch.load(max_step_file_unlearn)
        model_unlearn.load_state_dict(ckpt_unlearn["model"])
        model_unlearn.eval()

        x_t=torch.randn((args.n_samples, 1, config["image_size"], config["image_size"])).to(device)

        samples_original = model_full._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
        samples_ablated = model_ablated._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
        samples_unlearn = model_unlearn._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)

        # print(calculate_fid_mnist(samples_unlearn, samples_original), calculate_fid_mnist(samples_ablated, samples_original), calculate_fid_mnist(samples_ablated, samples_unlearn))

        print(clip_score(samples_unlearn, samples_original), clip_score(samples_ablated, samples_original), clip_score(samples_ablated, samples_unlearn))



if __name__=="__main__":
    args=parse_args()
    main(args)