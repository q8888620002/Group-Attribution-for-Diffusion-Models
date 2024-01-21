"""Calcuation D-TRAK, relative IF, randomized IF"""
import torch
import numpy as np
from attributions.attribution_utils import load_gradient_data

def compute_dtrak_trak_scores(args, train_idx, val_idx):
    """
    Compute scores for D-TRAK, TRAK, and influence function.
    """
    scores = np.zeros((len(val_idx), 1))

    # Iterating only through validation set for D-TRAK/TRAK.

    for i in train_idx:
        # obtain kernel from training subset

        dstore_keys = load_gradient_data(args, i)
        dstore_keys = torch.from_numpy(dstore_keys).cuda()

        print(dstore_keys.size())
        kernel = dstore_keys.T @ dstore_keys
        kernel = kernel + 5e-1 * torch.eye(kernel.shape[0]).cuda()

        kernel = torch.linalg.inv(kernel)

        print(kernel.shape)
        print(torch.mean(kernel.diagonal()))

    for i in val_idx:

        # Step2: calculate D-TRAK or TRAK for validation subsets
        # in https://github.com/sail-sg/d-trak.

        dstore_keys = load_gradient_data(args, i)
        dstore_keys = torch.from_numpy(dstore_keys).cuda()

        print(dstore_keys.size())

        score = dstore_keys @ ((dstore_keys @ kernel).T)
        print(score.size())

        # Normalize based on the meganitude. 

        if args.attribution_method == "relative_if":
            magnitude = np.linalg.norm(dstore_keys @ kernel)
        elif args.attribution_method == "randomized_if":
            magnitude = np.linalg.norm(dstore_keys)
        else:
            magnitude = 1

        scores[i] = score.cpu().numpy()/magnitude

    return scores