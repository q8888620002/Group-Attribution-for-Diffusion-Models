# Data Attribution - MNIST Diffusion
![60 epochs training from scratch](assets/demo.gif "60 epochs training from scratch")

## Training
Install packages
```bash
pip install -r requirements.txt
```
Start default setting training 
```bash
python train_mnist_unlearning.py --epochs=300 --loss_type="type2" --device="cuda:1"
```
Feel free to tuning training parameters, type `python train_mnist.py -h` to get help message of arguments.


## Reference
A neat blog explains how diffusion model works(must read!): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

A pytorch version of DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch

