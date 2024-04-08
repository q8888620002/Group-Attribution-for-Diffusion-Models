"""Prune LoRA weights of a text-to-image diffusion model."""
import argparse
import os

import numpy as np
import torch
import torch_pruning as tp
from tqdm import tqdm
from transformers import CLIPTextModel

from diffusers import DiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from src.utils import fix_get_processor, get_module, print_args


def parse_args():
    """Parsing arguments"""
    parser = argparse.ArgumentParser()
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
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Variant of the model files of the pretrained model identifier from "
            "huggingface.co/models, 'e.g.' fp16"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="the resolution for input images",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="directory containing LoRA weights to load",
        required=True,
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        help="ratio for remaining parameters",
        default=0.3,
    )
    return parser.parse_args()


def main(args):
    """Main function for pruning."""
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    unet = pipeline.unet.to("cuda")
    fix_get_processor(unet)

    # Figure out the hidden state shape.
    if isinstance(pipeline.text_encoder, CLIPTextModel):
        position_embedding = (
            pipeline.text_encoder.text_model.embeddings.position_embedding
        )
        hidden_state_shape = position_embedding.weight.shape
    else:
        text_encoder_type = type(pipeline.text_encoder)
        raise NotImplementedError(
            f"hidden state shape retrieval not implemeted for {text_encoder_type}"
        )

    # Figure out the latent sample shape.
    with torch.no_grad():
        latent = pipeline.vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution)
        ).latent_dist.sample()
        latent_shape = latent.shape

    example_inputs = {
        "sample": torch.randn(latent_shape).to("cuda"),
        "timestep": torch.ones((1,)).long().to("cuda"),
        "encoder_hidden_states": torch.randn(1, *hidden_state_shape).to("cuda"),
    }

    unet_macs, unet_params = tp.utils.count_ops_and_params(unet, example_inputs)

    unet.load_attn_procs(args.lora_dir)
    print(f"LoRA weights loaded from {args.lora_dir}")

    unet_lora_macs, unet_lora_params = tp.utils.count_ops_and_params(
        unet, example_inputs
    )
    unet.zero_grad()
    unet.eval()

    # Set up dependency graph and importance metric.
    dg = tp.DependencyGraph().build_dependency(unet, example_inputs=example_inputs)
    imp = tp.importance.MagnitudeImportance()

    # Retrieve the LoRA modules.
    lora_dict = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinearLayer):
            lora_dict[name] = module

    # Calculate importance scores.
    print("Calculating importance score for each LoRA node...")
    node_list, node_score_list, node_param_size_list = [], [], []
    for name, lora in tqdm(lora_dict.items()):
        group_idxs = [i for i in range(lora.rank)]

        down = dg.get_pruning_group(
            lora.down, tp.prune_linear_out_channels, idxs=group_idxs
        )
        down_scores = imp(down).detach().cpu().tolist()
        down_nodes = [(name, i, "down") for i in range(lora.rank)]
        node_list.extend(down_nodes)
        node_score_list.extend(down_scores)
        node_param_size_list.extend([lora.down.in_features] * lora.rank)

        up = dg.get_pruning_group(lora.up, tp.prune_linear_in_channels, idxs=group_idxs)
        up_scores = imp(up).detach().cpu().tolist()
        up_nodes = [(name, i, "up") for i in range(lora.rank)]
        node_list.extend(up_nodes)
        node_score_list.extend(up_scores)
        node_param_size_list.extend([lora.up.out_features] * lora.rank)

    # Identify the pairs of LoRA downsampling and upsampling nodes to remove. Pairs of
    # nodes, instead of individual nodes, are removed to ensure that the dependency
    # graph is still structurally sound.
    assert sum(node_param_size_list) == unet_lora_params - unet_params
    lora_param_size = sum(node_param_size_list)
    target_param_size = args.pruning_ratio * lora_param_size
    removed_pair_set = set()

    sorted_indices = np.argsort(node_score_list, kind="stable")
    for i in sorted_indices:
        node, node_param_size = node_list[i], node_param_size_list[i]
        pair = (node[0], node[1])  # (module name, group idx).
        pair_param_size = node_param_size * 2

        if pair not in removed_pair_set:
            removed_pair_set.add(pair)
            lora_param_size -= pair_param_size

        if lora_param_size <= target_param_size:
            break

    removed_module_dict = {}
    for pair in removed_pair_set:
        name, group_idx = pair
        if name not in removed_module_dict.keys():
            removed_module_dict[name] = []
        removed_module_dict[name].append(group_idx)

    # Prune the LoRA nodes.
    print("Pruning the LoRA weights...")
    for name, removed_idx_list in removed_module_dict.items():
        lora = get_module(unet, name)
        if len(removed_idx_list) > 0:
            tp.prune_linear_out_channels(lora.down, idxs=removed_idx_list)
            tp.prune_linear_in_channels(lora.up, idxs=removed_idx_list)
            lora.rank = lora.rank - len(removed_idx_list)
            assert lora.rank == lora.down.out_features
            assert lora.rank == lora.up.in_features

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(unet, example_inputs)
    lora_params = unet_lora_params - unet_params
    pruned_lora_params = pruned_params - unet_params
    actual_pruning_ratio = pruned_lora_params / lora_params

    # Save the pruned LoRA weights and pruning info.
    parsed_dir_list = args.lora_dir.split("/")
    if parsed_dir_list[-1] == "":
        parsed_dir_list = parsed_dir_list[:-1]

    # Replace the "method" part of the directory.
    parsed_dir_list[-3] = f"pruned_ratio={args.pruning_ratio}"
    outdir = "/".join(parsed_dir_list)

    unet.save_attn_procs(outdir)
    print(f"Pruned LoRA weights saved to {outdir}")

    info_file = os.path.join(outdir, "info.csv")
    with open(info_file, "w") as handle:
        handle.write("metric,value" + "\n")
        handle.write(f"unet_macs,{unet_macs:.0f}" + "\n")
        handle.write(f"unet_lora_macs,{unet_lora_macs:.0f}" + "\n")
        handle.write(f"pruned_unet_lora_macs,{pruned_macs:.0f}" + "\n")
        handle.write(f"unet_params,{unet_params:.0f}" + "\n")
        handle.write(f"lora_params,{lora_params:.0f}" + "\n")
        handle.write(f"pruned_lora_params,{pruned_lora_params:.0f}" + "\n")
        handle.write(f"target_pruning_ratio,{args.pruning_ratio}" + "\n")
        handle.write(f"actual_pruning_ratio,{actual_pruning_ratio:.5f}" + "\n")
    print(f"Pruning information saved to {info_file}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Pruning done!")
