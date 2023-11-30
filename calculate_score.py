import argparse
import torch
import numpy as np
import torchvision
import pickle
import torchvision.transforms.functional as TF
import os


from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from ddpm_config import DDPMConfig
from diffusion.model_util import create_ddpm_model
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


    parser.add_argument('--loss_type',type = str,help = 'define loss type',default='type4')
    parser.add_argument('--alpha1',type = float,help = 'loss params: alpha1',default=1)
    parser.add_argument('--alpha2',type = float,help = 'loss params: alpha2',default=1e-1)

    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--device', type= str , help = 'device to train')
    parser.add_argument('--weight_reg', action='store_true', help = 'whether to use weight as regularization.')
    parser.add_argument('--fine_tune_att', action='store_true', help = 'whether to fine tune only attentiona layers.')

    args = parser.parse_args()

    return args

def save_images(samples, directory, name_prefix):
    """

    Helpler function to save images

    """
    for i, img in enumerate(samples):

        save_image(
            img,
            os.path.join(directory, f"{name_prefix+i}.png")
        )

def main(args):

    ## EMA params
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)

    # Model params
    device = args.device
    batch_size = args.batch_size

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset  == "mnist":
        config = {**DDPMConfig.mnist_config}

    if config is None:
        raise ValueError(f"Invalid dataset: {args.dataset}")


    results = {
        "FID": [],
        'clip_score_unlearn':[],
        'pred_probs' : []
    }


    for target_digit in range(0, 10):

        model_unlearn =  create_ddpm_model(config).to(device)
        model_retrain =  create_ddpm_model(config).to(device)

        ## Load unlearn model

        path = f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/unlearning/"
        params = f"models/{target_digit}/epochs={args.epochs}_lr={args.lr}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}_weight_reg={args.weight_reg}_fine_tune_att={args.fine_tune_att}/"

        max_steps_unlearn_file = get_max_step_file(path+params)
        ckpt_unlearn = torch.load(max_steps_unlearn_file)

        model_unlearn_ema = ExponentialMovingAverage(model_unlearn, device=device, decay=1.0 - alpha)
        model_unlearn_ema.load_state_dict(ckpt_unlearn["model_ema"])
        model_unlearn_ema.eval()

        ## Load retrain model

        path_retrain = f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{target_digit}/"

        max_steps_retrain_file = get_max_step_file(path_retrain)
        ckpt_retrain = torch.load(max_steps_retrain_file)

        model_retrain_ema = ExponentialMovingAverage(model_retrain, device=device, decay=1.0 - alpha)
        model_retrain_ema.load_state_dict(ckpt_retrain["model_ema"])
        model_retrain_ema.eval()

        if args.dataset == "mnist":

            ## TODO mnist evaluation

            cnn = CNN().to(device)
            cnn.load_state_dict(torch.load('eval/models/epochs=10_cnn_weights.pt'))
            cnn.eval()

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

        elif args.dataset == "cifar":

            # Define the base directories for saving the files
            unlearn_dir = os.path.join("/projects/leelab/mingyulu/data_att/results", args.dataset, "unlearning", "eval", str(target_digit))
            retrain_dir = os.path.join("/projects/leelab/mingyulu/data_att/results", args.dataset, "retrain", "eval", str(target_digit))

            os.makedirs(unlearn_dir, exist_ok=True)
            os.makedirs(retrain_dir, exist_ok=True)

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

            unlearn_features = []
            retrain_features = []

            n_batches = args.n_samples // batch_size
            count = 0

            with torch.no_grad():
                for _ in range(n_batches):

                    samples = model_unlearn_ema.module.sampling(
                        batch_size,
                        clipped_reverse_diffusion=not args.no_clip,
                        device=device
                    )
                    samples = torch.stack([TF.resize(sample, (299, 299), antialias=True) for sample in samples])

                    save_images(torch.clamp((samples+1.)/2, 0., 1.), unlearn_dir, count*batch_size)

                    unlearn_feat = inception(samples)[0]
                    unlearn_feat = unlearn_feat.squeeze(3).squeeze(2).cpu().numpy()
                    unlearn_features.append(unlearn_feat)

                    samples = model_retrain_ema.module.sampling(
                        batch_size,
                        clipped_reverse_diffusion=not args.no_clip,
                        device=device
                    )
                    samples = torch.stack([TF.resize(sample, (299, 299), antialias=True) for sample in samples])

                    save_images(torch.clamp((samples+1.)/2, 0., 1.), retrain_dir, count*batch_size)

                    retrain_feat = inception(samples)[0]
                    retrain_feat = retrain_feat.squeeze(3).squeeze(2).cpu().numpy()
                    retrain_features.append(retrain_feat)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    count += 1

            unlearn_features = np.concatenate(unlearn_features, axis=0)
            retrain_features = np.concatenate(retrain_features, axis=0)

            # Calculate FID

            fid_value1 = calculate_fid(real_features, unlearn_features)
            print('real vs unlearn: FID:', fid_value1)

            fid_value2 = calculate_fid(real_features, retrain_features)
            print('real vs retrain FID:', fid_value2)

            fid_value3 = calculate_fid(retrain_features, unlearn_features)

            print('retrain vs unlearn: FID:', fid_value3)

            results["FID"].append((fid_value1, fid_value2, fid_value3))

    # print(np.mean(results['pred_probs'], axis=0)[0], np.mean(results['pred_probs'], axis=0)[1])
    # print(np.mean(results['clip_score_unlearn']))

    with open('results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

if __name__=="__main__":
    args=parse_args()
    main(args)