"""
Compute inception score adapted from [1]

[1]: https://github.com/sbarratt/inception-score-pytorch
"""
import numpy as np
import torch
import torch.utils.data
from pytorch_fid.inception import _inception_v3
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F


def eval_is(images_dataset, batch_size=32, resize=False, normalize=False, splits=1):
    """
    Computes the inception score of the generated images imgs

    Args:
    ----
        images_dataset: Torch dataset of (3xHxW) numpy images
        batch_size: batch size for feeding into Inception v3
        resize: whether resize images to 299 x 299
        normalize: thether to normalize the image to scale [-1, 1]
        splits: number of splits
    Return:
    ------
        Inception scores
    """

    assert batch_size > 0

    # Set up dtype

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    dataloader = torch.utils.data.DataLoader(images_dataset, batch_size=batch_size)

    inceptionNet = _inception_v3(pretrained=True, transform_input=False).type(dtype)
    inceptionNet.eval()  # Important: .eval() is needed to turn off dropout

    up = nn.Upsample(size=(299, 299), mode="bilinear").type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        x = inceptionNet(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    N = len(images_dataset)
    preds = np.zeros((N, 1000))

    for i, data in enumerate(dataloader):
        data = data.type(dtype)
        batch_size_i = len(data)
        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(data)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)
