import os
import argparse
import numpy as np
import torch

from diffusion.diffusions import DDPM
from diffusion.models import CNN
from ddpm_config import DDPMConfig

from utils import *



def parse_args():

    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    # Training params
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--device', type=str,help = 'gpu training', default="cuda:0")
    parser.add_argument('--batch_size',type = int ,default=128)
    parser.add_argument('--lr',type = float ,default=1e-3)

    ## Diffusion params

    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    ## Loss related params

    parser.add_argument('--loss_type',type = str,help = 'define loss type',default='type1')
    parser.add_argument('--alpha1',type = float,help = 'loss params: alpha1',default=1)
    parser.add_argument('--alpha2',type = float,help = 'loss params: alpha2',default=1e-1)
    parser.add_argument('--weight_reg', action='store_true', help = 'whether to use weight as regularization.')

    args = parser.parse_args()

    return args

def main(args):

    ## Load CNN

    device = args.device

    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load('eval/models/epochs=10_cnn_weights.pt'))
    cnn.eval()

    ## loss params

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    #torchvision ema setting - https://github.com/pytorch/vision/blob/main/references/classification/train.py#

    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)


    ## load diffusion

    results = []

    for target_digit in range(1, 10):

        path = f"/projects/leelab/mingyulu/data_att/results/mnist/unlearning/"
        params = f"models/{target_digit}/epochs={args.epochs}_lr={args.lr}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}_weight_reg={args.weight_reg}/"
        max_steps_file = get_max_step_file(path+params)

        config = {**DDPMConfig.mnist_config}

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
            dropout=config['dropout'],
        ).to(device)

        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        ckpt=torch.load(max_steps_file)

        model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])

        n_batches = args.n_samples // args.batch_size

        probs = []

        for _ in range(n_batches):
            with torch.no_grad():

                samples = model_ema.module.sampling(
                    args.batch_size,
                    clipped_reverse_diffusion=not args.no_clip,
                    device=device
                )

                outputs = torch.nn.functional.softmax(cnn(samples)[0], dim=1)
                prob = outputs[:, target_digit].detach().cpu().numpy()
                probs.append(prob)

        print(f" predicted probabilitys for {target_digit}, {np.mean(probs)}")

    print(f" final results:,", results.mean())


if __name__== "__main__":
    args=parse_args()
    main(args)


