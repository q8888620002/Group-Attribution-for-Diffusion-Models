# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File adapted from https://github.com/huggingface/diffusers/blob/v0.24.0-release/examples/text_to_image/train_text_to_image_lora.py
"""LoRA Gradient computation script for Stable Diffusion LoRA."""

# Not using the optimized dot product operation because vmap does not work for it.
# It's important that this attribute is deleted at the very top to ensure it's not used.
import torch.nn.functional as F
delattr(F, "scaled_dot_product_attention")

import argparse
import logging
import math
import os
import sys
import random
import shutil
from pathlib import Path

import datasets
import diffusers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import time
from src.datasets import remove_data_by_shapley, remove_data_by_uniform, remove_data_by_datamodel
from src.ddpm_config import PromptConfig, LoraUnlearningConfig, LoraSparseUnlearningConfig
from src.utils import fix_get_processor
from trak.projectors import ProjectionType, CudaProjector

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/miniSD-diffusers",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="train",
        choices=["train", "generated", "generated_journey"],
        help="source of data for computing gradients",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--num_images", type=int, default=50, help="number of generated images",
    )
    parser.add_argument(
        "--generation_seed", type=int, default=42, help="seed for image generation",
    )
    parser.add_argument(
        "--num_journey_points",
        type=int,
        default=50,
        help="number of time points selected for Journey-TRAK",
    )
    parser.add_argument(
        "--num_journey_noises",
        type=int,
        default=1,
        help="number of noises to sample for intermediate latent at each time step",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--cls_key",
        type=str,
        default="style",
        help="dataset key for class labels",
    )
    parser.add_argument(
        "--cls",
        type=str,
        default="post_impressionism",
        help="fine-tune only on a specific class in the dataset",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        help="directory containing trained LoRA weights to load",
        default=None,
    )
    parser.add_argument(
        "--lora_steps",
        type=int,
        help="number of trained steps for the LoRA weights to load",
        default=None,
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="number of timesteps to select for computing gradients",
        default=100,
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        help="dimension size for projected gradients",
        default=32768,
    )
    parser.add_argument(
        "--f",
        type=str,
        help="loss function for computing gradients",
        required=True,
    )

    args = parser.parse_args()

    # Sanity checks
    if args.train_data_dir is None:
        raise ValueError("Need a training folder.")
    assert not hasattr(F, "scaled_dot_product_attention")

    return args

def count_parameters(model):
    """Count the number of parameters requiring gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    args.dataset = (
        "artbench" if "artbench" in args.train_data_dir else args.train_data_dir
    )
    if args.cls is not None and args.cls_key is not None:
        args.dataset = args.dataset + f"_{args.cls}"

    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.dataset, "gradients", args.source)
        os.makedirs(args.output_dir, exist_ok=True)

    # If passed along, set the seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16

    # Move vae and text_encoder to device and cast to weight_dtype.
    vae.to("cuda", dtype=weight_dtype)
    text_encoder.to("cuda", dtype=weight_dtype)

    # Load LoRA weights, with runtime bugfix when LoRA ranks are different across 
    # attention to_q, to_k, to_v, and to_out.
    fix_get_processor(unet)
    weight_name = "pytorch_lora_weights"
    if args.lora_steps is not None:
        weight_name += f"_{args.lora_steps}"
    weight_name += ".safetensors"

    unet.load_attn_procs(args.lora_dir, weight_name=weight_name)
    lora_file = os.path.join(args.lora_dir, weight_name)
    print(f"LoRA weights loaded from {lora_file}")

    # Convert non-LoRA parameters to the specified precision.
    unet_state_dict = unet.state_dict()
    for name, param in unet_state_dict.items():
        if "lora_layer" not in name:
            unet_state_dict[name] = unet_state_dict[name].to(weight_dtype)
    unet.load_state_dict(unet_state_dict, assign=True)
    unet.requires_grad_(False)
    unet.to("cuda")

    # Require gradients for LoRA weights.
    for name, param in unet.named_parameters():
        if "lora_layer" in name:
            param.requires_grad_(True)

    if args.source == "train":
        # Get the training dataset.
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        if args.cls is not None and args.cls_key is not None:
            cls_idx = np.where(np.array(dataset["train"][args.cls_key]) == args.cls)[0]
            dataset["train"] = dataset["train"].select(cls_idx)
            if "artbench" in args.dataset:
                assert dataset["train"].num_rows == 5000

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        image_column, caption_column = column_names[0], column_names[1]

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs.input_ids

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples


        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,  # Do not turn on shuffle to keep the group mapping intact!
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )

        # Get training data latents and text encoder hidden states.
        all_latents, all_encoder_hidden_states = [], []
        for batch in train_dataloader:
            for key in batch.keys():
                batch[key] = batch[key].to("cuda")

            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.mode()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            all_latents.append(latents.detach().cpu())
            all_encoder_hidden_states.append(encoder_hidden_states.detach().cpu())
        all_latents = torch.cat(all_latents)
        all_encoder_hidden_states = torch.cat(all_encoder_hidden_states)

        group_df = pd.DataFrame(
            {
                "index": [i for i in range(dataset["train"].num_rows)],
                "artist": dataset["train"]["artist"],
                "filename": dataset["train"]["filename"],
            }
        )
        group_df.to_csv(os.path.join(args.output_dir, "group.csv"), index=False)
    else:
        if "artbench" in args.dataset:
            prompt_dict = PromptConfig.artbench_config
        else:
            raise NotImplementedError
        assert args.cls is not None
        prompt = prompt_dict[args.cls]
        input_ids = tokenizer(
            [prompt],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids.to("cuda"))[0]
        
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to("cuda")

        weight_name = "pytorch_lora_weights"
        if args.lora_steps is not None:
            weight_name += f"_{args.lora_steps}"
        weight_name += ".safetensors"
        pipeline.unet.load_attn_procs(args.lora_dir, weight_name=weight_name)
        weight_path = os.path.join(args.lora_dir, weight_name)
        print(f"LoRA weights loaded onto the pipeline from {weight_path}")

        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(args.generation_seed)

        all_step_idx, all_t, all_latents, all_generated_image_idx = [], [], [], []
        print("Obtaining latents for generated images...")
        for i in tqdm(range(args.num_images)):
            step_idx_list, t_list, latents_list = [], [], []
            def extract_latents(step_idx, t, latents):
                step_idx_list.append(step_idx)
                t_list.append(t.detach().cpu())
                latents_list.append(latents.detach().cpu())

            _ = pipeline(
                prompt,
                num_inference_steps=100,
                generator=generator,
                height=args.resolution,
                width=args.resolution,
                callback=extract_latents,
                callback_steps=1,
            ).images[0]
            generated_image_idx_list = [i] * len(step_idx_list)
            
            if args.source == "generated":
                # Collect only the final latent variable.
                all_step_idx.append(step_idx_list[-1])
                all_t.append(t_list[-1])
                all_latents.append(latents_list[-1])
                all_generated_image_idx.append(generated_image_idx_list[-1])

            elif args.source == "generated_journey":
                num_inference_steps = len(step_idx_list)
                for j in np.arange(
                    start=1,
                    stop=num_inference_steps,
                    step=num_inference_steps // args.num_journey_points,
                ):
                    all_step_idx.append(step_idx_list[j])
                    all_t.append(t_list[j])
                    all_latents.append(latents_list[j])
                    all_generated_image_idx.append(generated_image_idx_list[j])
            else:
                raise NotImplementedError
        
        group_df = pd.DataFrame(
            {"generated_image_idx": all_generated_image_idx, "step_idx": all_step_idx}
        )
        group_df.to_csv(os.path.join(args.output_dir, "group.csv"), index=True)

        all_latents = torch.cat(all_latents).to(weight_dtype)
        all_t = torch.stack(all_t)
        all_encoder_hidden_states = []
        for _ in range(all_latents.size(0)):
            all_encoder_hidden_states.append(encoder_hidden_states.detach().cpu().clone())
        all_encoder_hidden_states = torch.cat(all_encoder_hidden_states)
    
    if args.source == "generated_journey":
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_latents, all_t, all_encoder_hidden_states),
            shuffle=False,  # Do not turn on shuffle to keep the group mapping intact!
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_latents, all_encoder_hidden_states),
            shuffle=False,  # Do not turn on shuffle to keep the group mapping intact!
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )

    unet.eval()

    projector = CudaProjector(
        grad_dim=count_parameters(unet),
        proj_dim=args.proj_dim,
        seed=42,
        proj_type=ProjectionType.normal,
        device="cuda",
        max_batch_size=args.train_batch_size,
    )

    params = {k: v.detach() for k, v in unet.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in unet.named_buffers() if v.requires_grad}

    from torch.func import functional_call, vmap, grad 
    def vectorize_and_ignore_buffers(g, params_dict=None):
        """
        gradients are given as a tuple :code:`(grad_w0, grad_w1, ... grad_wp)` where
        :code:`p` is the number of weight matrices. each :code:`grad_wi` has shape
        :code:`[batch_size, ...]` this function flattens :code:`g` to have shape
        :code:`[batch_size, num_params]`.
        """
        batch_size = len(g[0])
        out = []
        if params_dict is not None:
            for b in range(batch_size):
                out.append(torch.cat([x[b].flatten() for i, x in enumerate(g) if is_not_buffer(i, params_dict)]))
        else:
            for b in range(batch_size):
                out.append(torch.cat([x[b].flatten() for x in g]))
        return torch.stack(out)

    if args.f=='mean-squared-l2-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            # predictions = predictions.reshape(1, -1)
            # f = torch.norm(predictions.float(), p=2.0, dim=-1)**2 # squared
            # f = f/predictions.size(1) # mean
            # f = f.mean()
            ####
            f = F.mse_loss(predictions.float(), torch.zeros_like(targets).float(), reduction="none")
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='mean':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            f = predictions.float()
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='l1-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=1.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='l2-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=2.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    elif args.f=='linf-norm':
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=float('inf'), dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f
    else:
        print(args.f)
        def compute_f(params, buffers, noisy_latents, timesteps, encoder_hidden_states, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            targets = targets.unsqueeze(0)
   
            predictions = functional_call(unet, (params, buffers), args=noisy_latents, 
                                  kwargs={'timestep': timesteps, 
                                          'encoder_hidden_states': encoder_hidden_states
                                         })
            predictions = predictions.sample
            ####
            f = F.mse_loss(predictions.float(), targets.float(), reduction="none")
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            return f
    
    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0,))

    if args.source == "generated_journey":
        all_embs = []
        for (latents, timesteps, encoder_hidden_states) in tqdm(dataloader):
            latents = latents.to("cuda")
            timesteps = timesteps.to("cuda")
            encoder_hidden_states = encoder_hidden_states.to("cuda")

            for index_noise in range(args.num_journey_noises):    
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                ft_per_sample_grads = ft_compute_sample_grad(
                    params,
                    buffers,
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    target,
                )

                ft_per_sample_grads = vectorize_and_ignore_buffers(
                    list(ft_per_sample_grads.values())
                )
                if index_noise == 0:
                    emb = ft_per_sample_grads
                else:
                    emb += ft_per_sample_grads
            emb = emb / args.num_journey_noises
            emb = projector.project(emb, model_id=0)
            all_embs.append(emb.detach().cpu())

        all_embs = torch.cat(all_embs)
        output_filename = f"emb_f={args.f}_num_journey_points={args.num_journey_points}_num_journey_noises={args.num_journey_noises}_proj_dim={args.proj_dim}.pt"
        torch.save(all_embs, os.path.join(args.output_dir, output_filename))
    else:
        all_embs = []
        for (latents, encoder_hidden_states) in tqdm(dataloader):
            latents = latents.to("cuda")
            encoder_hidden_states = encoder_hidden_states.to("cuda")

            bsz = latents.shape[0]
            selected_timesteps = range(0, 1000, 1000 // args.num_timesteps)

            for index_t, t in enumerate(selected_timesteps):
                timesteps = torch.tensor([t]*bsz, device=latents.device)
                timesteps = timesteps.long()
                        
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                ft_per_sample_grads = ft_compute_sample_grad(
                    params,
                    buffers,
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    target,
                )

                ft_per_sample_grads = vectorize_and_ignore_buffers(
                    list(ft_per_sample_grads.values())
                )  
                
                if index_t==0:
                    emb = ft_per_sample_grads
                else:
                    emb += ft_per_sample_grads
            emb = emb / args.num_timesteps
            emb = projector.project(emb, model_id=0)
            all_embs.append(emb.detach().cpu())

        all_embs = torch.cat(all_embs)
        output_filename = f"emb_f={args.f}_num_timesteps={args.num_timesteps}_proj_dim={args.proj_dim}.pt"
        torch.save(all_embs, os.path.join(args.output_dir, output_filename))


if __name__ == "__main__":
    main()
    print("Done!")
