"""Functions for calculating aesthetics score for an image."""

import os
import torch
import torch.nn as nn
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    cache_folder = os.path.join(os.environ["XDG_CACHE_HOME"], "emb_reader")
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m
