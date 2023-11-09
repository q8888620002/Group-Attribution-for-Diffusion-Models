import torch.nn as nn
import torch
import math

from diffusion.models import UNet
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(
            self,
            timesteps=1000,
            base_dim=128,
            channel_mult=[1, 2, 3, 4],
            image_size:int=32,
            in_channels:int=3,
            out_channels:int=3
        ):

        super().__init__()

        self.timesteps=timesteps
        self.image_size=image_size
        self.in_channels = in_channels

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model = UNet(
            T=timesteps,
            ch=base_dim,
            ch_mult=channel_mult,
            attn=[2],
            num_res_blocks=2,
            dropout=0.1,
            input_ch_dim = in_channels,
            output_ch_dim = out_channels
        )


    def forward(
        self,
        x: torch.tensor,
        noise: torch.tensor,
        t: torch.tensor = None
    )-> torch.tensor:
        # x:NCHW

        if t is None:
            ## random t
            t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)

        x_t=self._forward_diffusion(x,t,noise)

        ## pred noise from U-Net

        pred_noise=self.model(x_t,t)

        return pred_noise

    @torch.no_grad()
    def sampling(
            self,
            n_samples: int,
            clipped_reverse_diffusion: bool=True,
            device: str="cuda"
    ) ->  torch.tensor:
        """
        Sampling process for the model x_T -> x_0
        """

        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)

        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):

            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t

    @torch.no_grad()
    def _sampling(
        self,
        x_t,
        n_samples,
        clipped_reverse_diffusion=True,
        device="cuda"
    )-> torch.tensor:
        """
        Sampling process for the model x_T -> x_0
        """
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):

            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t

    def _cosine_variance_schedule(
            self,
            timesteps: int,
            epsilon: float= 0.008
        )-> float:

        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps], 0.0 , 0.999 )

        return betas

    def _forward_diffusion(
            self,
            x_0: torch.tensor,
            t: torch.tensor,
            noise: torch.tensor
        )-> torch.tensor:

        """
        Forward process q(x_{t}|x_{t-1})
        """

        assert x_0.shape == noise.shape


        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(
            self,
            x_t: torch.tensor,
            t: torch.tensor,
            noise: torch.tensor
        ) -> torch.tensor:
        '''
        Reverse process: p(x_{t-1}|x_{t})-> mean,std
                        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)

        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise


    @torch.no_grad()
    def _reverse_diffusion_with_clip(
            self,
            x_t: torch.tensor,
            t: torch.tensor,
            noise: torch.tensor
        ) -> torch.tensor:
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)

        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise