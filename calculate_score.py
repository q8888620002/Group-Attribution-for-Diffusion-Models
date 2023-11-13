import argparse
import torch
import numpy as np
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

from ddpm_config import DDPMConfig
from diffusion.diffusions import DDPM
from utils import *
from data_attribution.eval.inception import InceptionV3

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

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset  == "mnist":
        config = {**DDPMConfig.mnist_config}

    if config is None:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    model_full =  DDPM(
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
        ).to(device).to(device)
    
    # model_ablated = DDPM(**config).to(device)
    # model_unlearn = DDPM(**config).to(device)


    if args.dataset == "mnist":

        ## TODO mnist evaluation

        max_step_file = find_max_step_file("/projects/leelab2/mingyulu/unlearning/results/full/models/steps_*.pt")
        ckpt = torch.load(max_step_file)

        # model_full.load_state_dict(ckpt["model"])
        # model_full.eval()

        # for i in range(1, 10):
        #     max_step_file_ablated = find_max_step_file(f"/projects/leelab2/mingyulu/unlearning/results/ablated/{i}/models/steps_*.pt")
        #     ckpt_ablated = torch.load(max_step_file_ablated)
        #     model_ablated.load_state_dict(ckpt_ablated["model"])
        #     model_ablated.eval()

        #     max_step_file_unlearn = find_max_step_file(f"results/models/unlearn_remaining_ablated/{i}/epochs=100_datasets=False_loss=type1:alpha1=1.0_alpha2=0.01_weight_reg=False/steps_*.pt")
        #     ckpt_unlearn = torch.load(max_step_file_unlearn)
        #     model_unlearn.load_state_dict(ckpt_unlearn["model"])
        #     model_unlearn.eval()

        #     x_t=torch.randn((args.n_samples, 1, config["image_size"], config["image_size"])).to(device)

        #     samples_original = model_full._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
        #     samples_ablated = model_ablated._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
        #     samples_unlearn = model_unlearn._sampling(x_t, args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)


            # print(clip_score(samples_unlearn, samples_original), clip_score(samples_ablated, samples_original), clip_score(samples_ablated, samples_unlearn))

    elif args.dataset == "cifar":

        # max_step_file = find_max_step_file("results/cifar/retrain/models/full/steps_*.pt")

        ckpt = torch.load("/projects/leelab/mingyulu/data_att/results/cifar/retrain/models/full/steps_00125000.pt")

        # ckpt = torch.load("results/cifar/retrain/models/full/steps_00078200.pt")

        model_full.load_state_dict(ckpt["model"])
        model_full.eval()

        transform = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=config['mean'], std=config['std'])
            ]
        )

        mean = torch.tensor(config['mean']).view( -1, 1, 1).to(device)
        std = torch.tensor(config['std']).view( -1, 1, 1).to(device)

        trainset = torchvision.datasets.CIFAR10(
            root=f'/projects/leelab/mingyulu/data_att/{args.dataset}',
            train=True,
            download=True, 
            transform=transform,
        )
        batch_size = 128  

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) 
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

        inception = InceptionV3([block_idx]).to(device)

        real_features = get_features(
            trainloader, 
            inception, 
            args.n_samples,
            device
        )
        
        fake_features = []
        fake_features_norm = []

        n_batches = args.n_samples // batch_size

        with torch.no_grad(): 
            for _ in range(n_batches):

                samples = model_full.sampling(batch_size, clipped_reverse_diffusion=not args.no_clip, device=device)

                normalized_samples = torch.stack([TF.resize((sample - mean)/std, (299, 299), antialias=True) for sample in samples])

                samples = torch.stack([TF.resize(sample, (299, 299), antialias=True) for sample in samples])

                fake_feat = inception(samples)[0]
                fake_feat = fake_feat.squeeze(3).squeeze(2).cpu().numpy()
                fake_features.append(fake_feat)

                fake_feat_norm = inception(normalized_samples)[0]
                fake_feat_norm = fake_feat_norm.squeeze(3).squeeze(2).cpu().numpy()

                fake_features_norm.append(fake_feat_norm)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        fake_features = np.concatenate(fake_features, axis=0)
        fake_features_norm = np.concatenate(fake_features_norm, axis=0)

        # Calculate FID
        fid_value = calculate_fid(real_features, fake_features)
        print('FID:', fid_value)

        fid_value = calculate_fid(real_features, fake_features_norm)
        print('FID (norm):', fid_value)


if __name__=="__main__":
    args=parse_args()
    main(args)