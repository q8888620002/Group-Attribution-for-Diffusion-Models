import os
import math
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
import torchvision
from scipy.linalg import sqrtm
from scipy import linalg

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize, ToTensor, Compose, Lambda, Grayscale
from scipy.linalg import sqrtm


from diffusion.diffusions import DDPM
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

def calculate_fid(real_features, fake_features):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Feature extraction
def get_features(dataloader, model, n_samples):

    
    model.eval()

    with torch.no_grad():

        images, _ = next(iter(dataloader))
        images = images.to(device)[:n_samples]
        images = torch.stack([TF.resize(img, (299, 299), antialias=True) for img in images])

        features = model(images)
    return features.cpu().numpy()

def main(args):

    dataset_configs = {
        "mnist": {
            "timesteps":1000,
            "base_dim": 64,
            "image_size": 28,
            "in_channels": 1,
            "out_channels":1,
            "channel_mult": [2, 4],
        },
        "cifar": {
            "timesteps":1000,
            "base_dim": 128,
            "channel_mult": [1, 2, 3, 4],
            "image_size": 32,
            "in_channels": 3,
            "out_channels": 3,
        },
    }



    config = dataset_configs.get(args.dataset)

    if config is None:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    model_full = DDPM(**config).to(device)
    model_ablated = DDPM(**config).to(device)
    model_unlearn = DDPM(**config).to(device)

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


            print(clip_score(samples_unlearn, samples_original), clip_score(samples_ablated, samples_original), clip_score(samples_ablated, samples_unlearn))

    if args.dataset == "cifar":

        # max_step_file = find_max_step_file("results/cifar/retrain/models/full/steps_*.pt")
        # import ipdb;ipdb.set_trace()

        ckpt = torch.load("results/cifar/retrain/models/full/steps_00078200.pt")

        model_full.load_state_dict(ckpt["model"])
        model_full.eval()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root='/data2/mingyulu/cfair_10', 
            train=True,
            download=True, 
            transform=transform,
        )

        inception_model = inception_v3(
            weights='DEFAULT',
            transform_input=False,
            aux_logits=True
        ).to(device)
        
        inception_model.to(device)
        inception_model.eval()

        batch_size = 128  

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) 

        # Calculate real image features
        real_features = get_features(trainloader, inception_model, n_samples=args.n_samples)


        fake_features = []

        n_batches = args.n_samples // batch_size
        
        with torch.no_grad(): 
            for _ in range(n_batches):
                x_t = torch.randn((batch_size, config["in_channels"], config["image_size"], config["image_size"])).to(device)
                samples = model_full._sampling(x_t, batch_size, clipped_reverse_diffusion=not args.no_clip, device=device)
                samples = torch.stack([TF.resize(sample, (299, 299), antialias=True) for sample in samples])
            
                fake_feat = inception_model(samples).cpu().numpy()
                fake_features.append(fake_feat)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        fake_features = np.concatenate(fake_features, axis=0)

        # Calculate FID
        fid_value = calculate_fid(real_features, fake_features)
        print('FID:', fid_value)


if __name__=="__main__":
    args=parse_args()
    main(args)