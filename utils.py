import torch

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms 

#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)
        
        
def create_mnist_dataloaders(
        batch_size,image_size=28,
        num_workers=4,
        exclude_label=None
    ):
    
    """
    
    Create a mnist dataset with/without filtered labels
    
    Args:
        batch_size: int
        num_worksrs: int
        exclude_label: int e.g. 1,2 
    Return:
        MNIST dataset. 
        
    """
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess
        )
    test_dataset=MNIST(
        root="./mnist_data",
        train=False,
        download=True,
        transform=preprocess
        )

    if exclude_label is not None:
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label != exclude_label]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label != exclude_label]
        
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
