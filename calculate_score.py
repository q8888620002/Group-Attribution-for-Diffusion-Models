import argparse
import torch
import numpy as np
import torchvision
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from ddpm_config import DDPMConfig
from diffusion.diffusions import DDPM
from diffusion.models import CNN
from eval.inception import InceptionV3
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--dataset',type = str,help = 'dataset name',default='')


    parser.add_argument('--loss_type',type = str,help = 'define loss type',default='type1')
    parser.add_argument('--alpha1',type = float,help = 'loss params: alpha1',default=1)
    parser.add_argument('--alpha2',type = float,help = 'loss params: alpha2',default=1e-1)

    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--device', type= str , help = 'device to train')
    parser.add_argument('--weight_reg', action='store_true', help = 'whether to use weight as regularization.')

    args = parser.parse_args()

    return args


def main(args):

    ## EMA params
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)

    # Model params
    device = args.device

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset  == "mnist":
        config = {**DDPMConfig.mnist_config}

    if config is None:
        raise ValueError(f"Invalid dataset: {args.dataset}")


    if args.dataset == "mnist":

        ## TODO mnist evaluation

        results = {
            'clip_score_unlearn':[],
            'pred_probs' : []
        }

        for target_digit in range(1, 10):

            path = f"/projects/leelab/mingyulu/data_att/results/mnist/unlearning/"
            path_retrain = f"/projects/leelab/mingyulu/data_att/results/mnist/retrain/models/{target_digit}/"

            params = f"models/{target_digit}/epochs={args.epochs}_lr={args.lr}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}_weight_reg={args.weight_reg}/"

            cnn = CNN().to(device)
            cnn.load_state_dict(torch.load('eval/models/epochs=10_cnn_weights.pt'))
            cnn.eval()

            model_unlearn =  DDPM(
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

            model_retrain =  DDPM(
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

            ## Load unlearn model

            max_steps_unlearn_file = get_max_step_file(path+params)
            ckpt_unlearn = torch.load(max_steps_unlearn_file)
            model_unlearn_ema = ExponentialMovingAverage(model_unlearn, device=device, decay=1.0 - alpha)
            model_unlearn_ema.load_state_dict(ckpt_unlearn["model_ema"])
            model_unlearn_ema.eval()

            ## Load retrain model

            max_steps_retrain_file = get_max_step_file(path_retrain)
            ckpt_retrain = torch.load(max_steps_retrain_file)
            model_retrain_ema = ExponentialMovingAverage(model_retrain, device=device, decay=1.0 - alpha)
            model_retrain_ema.load_state_dict(ckpt_retrain["model_ema"])
            model_retrain_ema.eval()

            samples_unlearns = []
            samples_retrains = []

            pred_probs_unlearn = []
            pred_probs_retrain = []

            n_batches = args.n_samples // args.batch_size

            for _ in range(n_batches):

                with torch.no_grad():

                    x_t=torch.randn((
                        args.batch_size,
                        config["out_channels"],
                        config["image_size"],
                        config["image_size"]
                    )).to(device)

                    samples_unlearn = model_unlearn_ema.module._sampling(
                        x_t,
                        args.batch_size,
                        clipped_reverse_diffusion=not args.no_clip,
                        device=device
                    )

                    samples_retrain = model_retrain_ema.module._sampling(
                        x_t,
                        args.batch_size,
                        clipped_reverse_diffusion=not args.no_clip,
                        device=device
                    )

                    outputs = torch.nn.functional.softmax(cnn(samples_unlearn)[0], dim=1)
                    prob_unlearn = outputs[:, target_digit].detach().cpu().numpy()
                    pred_probs_unlearn.append(prob_unlearn)

                    outputs = torch.nn.functional.softmax(cnn(samples_retrain)[0], dim=1)
                    prob_retrain = outputs[:, target_digit].detach().cpu().numpy()
                    pred_probs_retrain.append(prob_retrain)

                    samples_unlearn = samples_unlearn.squeeze(1).detach().cpu().numpy()
                    samples_retrain = samples_retrain.squeeze(1).detach().cpu().numpy()

                    samples_unlearns.append(samples_unlearn)
                    samples_retrains.append(samples_retrain)

            samples_unlearns = np.concatenate(samples_unlearns, axis=0)
            samples_retrains = np.concatenate(samples_retrains, axis=0)

            result_score  = clip_score(samples_unlearns, samples_retrains)
            mean_pred_unlearn = np.mean(pred_probs_unlearn)
            mean_retrain = np.mean(pred_probs_retrain)

            print(f"CLIP score for {target_digit}: unlearn: {result_score}")
            print(f"Predicted probabilitys for {target_digit}, unlearn: {mean_pred_unlearn}; retrain: {mean_retrain}")

            results['clip_score_unlearn'].append(result_score)
            results['pred_probs'].append((mean_pred_unlearn, mean_retrain))

        print(np.mean(results['pred_probs'][0]), np.mean(results['pred_probs'][1]))
        print(np.mean(results['clip_score_unlearn']))

        with open('results.pkl', 'wb') as fp:
            pickle.dump(results, fp)


    elif args.dataset == "cifar":

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

        ## Feature ranges should be [-1,1 ] according to https://github.com/mseitzer/pytorch-fid/issues/3
        ## If input scale is within [0,1] set normalize_input=True

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception = InceptionV3([block_idx],normalize_input=False).to(device)

        real_features = get_features(
            trainloader,
            mean,
            std,
            inception,
            args.n_samples,
            device
        )

        fake_features = []

        n_batches = args.n_samples // batch_size

        with torch.no_grad():
            for _ in range(n_batches):

                samples = model_full.sampling(batch_size, clipped_reverse_diffusion=not args.no_clip, device=device)
                samples = torch.stack([TF.resize(sample, (299, 299), antialias=True) for sample in samples])

                fake_feat = inception(samples)[0]
                fake_feat = fake_feat.squeeze(3).squeeze(2).cpu().numpy()
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