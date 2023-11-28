import os
import math
import argparse
import torch
import torch.nn as nn
import itertools

from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# from lora_diffusion import  inject_trainable_lora_extended, save_lora_weight
from diffusion.model_util import create_ddpm_model

from utils import *
from ddpm_config import DDPMConfig


def parse_args():

    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    # Training params
    parser.add_argument('--lr',type = float ,default=1e-4)
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
    parser.add_argument('--fine_tune_att', action='store_true', help = 'whether to fine tune only attentiona layers.')
    parser.add_argument('--fine_tune_lora', action='store_true', help = 'whether to fine tune with LORA.')

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
        config = {**DDPMConfig.cifar_config}
    elif args.dataset  == "mnist":
        config = {**DDPMConfig.mnist_config}
    else:
        raise ValueError(f"Unknown dataset {config['dataset']}, choose 'cifar' or 'mnist'.")

    model_frozen = create_ddpm_model(config).to(device)

    ## loss params

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    #torchvision ema setting - https://github.com/pytorch/vision/blob/main/references/classification/train.py#

    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)


    ckpt=torch.load(config['trained_model'])

    # ckpt = load_checkpoint(f"results/{args.dataset}/retrain/models/full/")

    model_frozen.load_state_dict(ckpt["model"])
    model_frozen.eval()

    freezed_model_dict = ckpt["model"]

    ## Make sure parameters of frozen mdoel is freezed.

    for params in model_frozen.parameters():
        params.requires_grad=False


    for excluded_class in range(0, 10):

        path = f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/unlearning/"
        exp_settings = f"/{excluded_class}/epochs={args.epochs}_lr={args.lr}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}_weight_reg={args.weight_reg}_fine_tune_att={args.fine_tune_att}"

        train_r1_dataloader, train_r2_dataloader, forget_dataloader = create_dataloaders(
            dataset_name=args.dataset,
            batch_size=config['batch_size'],
            excluded_class=excluded_class,
            unlearning=True
        )

        ## Init new model for unlearning.

        model = create_ddpm_model(config).to(device)

        ckpt=torch.load(config['trained_model'])
        model.load_state_dict(ckpt["model"])


        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
        model_ema.load_state_dict(ckpt["model_ema"])

        if args.fine_tune_att:

            ## Fine tune only attention or resblock.
            att_blocks = {'input_blocks.1.1', 'input_blocks.2.1', 'output_blocks.3.1', 'output_blocks.4.1', 'output_blocks.5.1'}

            for name, parameter in model.named_parameters():
                if any(block in name for block in att_blocks):
                    print(f"parameter '{name}' will not be freezed.")
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False

        # elif args.fine_tune_lora:

        #     for parameter in model.parameters():
        #         parameter.requires_grad = False

        #     ## fine tune with LORA.
        #     ## turn off all of the gradients of unet, except for the trainable LoRA params.

        #     UNET_DEFAULT_TARGET_REPLACE = { "ResBlock"}
        #     #{"AttentionBlock", "ResBlock", "SiLU"}
        #     rank = 6

        #     unet_lora_params, _ = inject_trainable_lora_extended(
        #         model,
        #         target_replace_module = UNET_DEFAULT_TARGET_REPLACE,
        #         r=rank,
        #         # verbose=True
        #     )

        #     unet_lora_params_ema, _ = inject_trainable_lora_extended(
        #         model_ema.module,
        #         target_replace_module= UNET_DEFAULT_TARGET_REPLACE,
        #         r=rank,
        #         # verbose=True
        #     )

        model_oracle = create_ddpm_model(config).to(device)

        for params in model_oracle.parameters():
            params.requires_grad=False

        max_steps_orcale_file = get_max_step_file(f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/retrain/models/{excluded_class}/")

        model_oracle.load_state_dict(torch.load(max_steps_orcale_file)["model"])
        model_oracle.eval()

        optimizer=AdamW(
            itertools.chain(*unet_lora_params) if args.fine_tune_lora else model.parameters(),
            lr=args.lr
        )

        scheduler=OneCycleLR(
            optimizer,
            args.lr,
            total_steps=args.epochs*len(train_r1_dataloader),
            pct_start=0.25,
            anneal_strategy='cos'
        )

        loss_fn=nn.MSELoss(reduction='mean')

        global_steps=0

        for epoch in range(args.epochs):

            model.train()

            ## iterating thorugh the size of unlearn dataset.
            epoch_loss = 0
            oracle_total_loss = 0

            for j, ((image_r1, _), (image_r2, _), (image_f, _)) in enumerate(zip(train_r1_dataloader, train_r2_dataloader, forget_dataloader)):

                image_r1=image_r1.to(device)
                image_r2=image_r2.to(device)
                image_f=image_f.to(device)

                ## Sample random noise e_t

                noise=torch.randn_like(image_r1).to(device)

                t=torch.randint(0, args.timesteps,(noise.shape[0],)).to(device)

                # get scores for D_r and D_e from the frozen model

                with torch.no_grad():
                    eps_r_frozen = model_frozen(image_r1, noise, t)
                    eps_f_frozen = model_frozen(image_f, noise, t)
                    eps_r_oracle = model_oracle(image_r1, noise, t)

                # Scores from the fine-tunning model

                eps_r = model(image_r1, noise, t)

                # delta logP(D_r) - delta logP(D_e)

                if args.loss_type == "type1":
                    loss = loss_fn(eps_r, alpha1*eps_r_frozen - alpha2*eps_f_frozen)

                elif args.loss_type  == "type2":
                    loss = alpha1*loss_fn(eps_r, eps_r_frozen) - alpha2*loss_fn(eps_r, eps_f_frozen)

                elif args.loss_type  == "type3":
                    ## Training with oracle loss - for debugging purpose

                    loss = loss_fn(eps_r, eps_r_oracle)

                elif args.loss_type == "type4":
                    loss = alpha1*loss_fn(eps_r, noise) + loss_fn(eps_r, (eps_r_frozen - alpha2*eps_f_frozen))

                ## weight regularization lambda*(\hat_{\theta} - \theta)
                ## TODO fisher information

                # if args.weight_reg:
                #     for n, p in model.named_parameters():
                #         _loss = (p - freezed_model_dict[n].to(device)) ** 2
                #         loss += 0.5 * _loss.sum()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                with torch.no_grad():
                    oracle_loss = loss_fn(eps_r, eps_r_oracle).detach().cpu().item()
                ## Update learning rate

                if global_steps % args.model_ema_steps==0:
                    model_ema.update_parameters(model)

                global_steps+=1

                epoch_loss += loss.detach().cpu().item()
                oracle_total_loss += oracle_loss

                if j % args.log_freq == 0:
                    print(f"Epoch[{epoch+1}/{args.epochs}],Step[{j}/{len(train_r1_dataloader)}],loss:{loss.detach().cpu().item():.6f},lr:{scheduler.get_last_lr()[0]:.6f}")
                    print(f"Oracle loss {oracle_loss:.6f}")

            print(f"Epoch total loss: {epoch_loss/(j+1)} ")
            print(f"Epoch total loss: {oracle_total_loss/(j+1)} ")

            if (epoch+1)% (config['epochs'] // 10) == 0 or (epoch+1)% args.epochs==0 or (global_steps+1) == config['epochs']*len(train_r1_dataloader):

                model_ema.eval()

                os.makedirs(f"results/{args.dataset}/unlearning/samples", exist_ok=True)
                os.makedirs(f"results/{args.dataset}/unlearning/samples" + exp_settings, exist_ok=True)

                samples = model_ema.module.sampling(
                    args.n_samples,
                    clipped_reverse_diffusion=not args.no_clip,
                    device=device
                )

                samples= torch.clamp(((samples+1.)/2.), 0., 1.)

                save_image(
                    samples,
                    f"results/{args.dataset}/unlearning/samples" + exp_settings + f"/steps_{global_steps:0>8}.png",

                    nrow=int(math.sqrt(args.n_samples))
                )


        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }

        # if args.fine_tune_lora:
        #     save_lora_weight(model,  path + "models" + exp_settings +  f"/steps_{global_steps:0>8}_lora_weight.pt")
        #     save_lora_weight(model_ema,  path + "models" + exp_settings +  f"/steps_{global_steps:0>8}_ema_lora_weight.pt")

        os.makedirs(path + "models" + exp_settings, exist_ok=True)
        torch.save(ckpt, path + "models" + exp_settings +  f"/steps_{global_steps:0>8}.pt")

if __name__=="__main__":
    args=parse_args()
    main(args)