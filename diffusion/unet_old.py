import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups
    def forward(self,x):
        n,c,h,w=x.shape
        x=x.view(n,self.groups,c//self.groups,h,w) # group
        x=x.transpose(1,2).contiguous().view(n,-1,h,w) #shuffle
        
        return x

class ConvBnSiLu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super().__init__()
        self.module=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.SiLU(inplace=True))
    def forward(self,x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.branch1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
                                    nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1=nn.Sequential(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.BatchNorm2d(in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv2d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.BatchNorm2d(out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        t_emb=self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x=x+t_emb
  
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
    def forward(self,x,t=None):
        x_shortcut=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x_shortcut,t)
        x=self.conv1(x)

        return [x,x_shortcut]
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        # import ipdb;ipdb.set_trace()

        x=self.upsample(x)
        x=torch.cat([x,x_shortcut],dim=1)
        x=self.conv0(x)

        if t is not None:
            x=self.time_mlp(x,t)
        x=self.conv1(x)

        return x      

class AttnBlock(nn.Module):
    """
    Self-attention block with residual connection.
    """

    def __init__(self, in_channels, dropout_rate=0.0):
        super().__init__()

        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # Query, Key, and Value projections
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Normalize input
        h = self.norm(x)

        # Project to Q, K, V
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute attention scores
        b, c, height, w = q.shape
        q = q.reshape(b, c, height * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, height * w)                   # b, c, hw
        attn_scores = torch.bmm(q, k)                # b, hw, hw
        attn_scores = attn_scores * (c ** -0.5)      # Scale scores
        attn_scores = F.softmax(attn_scores, dim=2)  # Softmax over keys
        attn_scores = self.dropout(attn_scores)      # Apply dropout to attention scores

        # Attend to values
        v = v.reshape(b, c, height * w)
        attn_scores = attn_scores.permute(0, 2, 1)  # b, hw, hw
        h = torch.bmm(v, attn_scores)               # b, c, hw
        h = h.reshape(b, c, height, w)

        # Project output
        h = self.proj_out(h)

        # Add residual connection
        return x + h
    
class Modified_Unet(nn.Module):
    '''
    Modified U-Net architecture with attention blocks.
    '''
    def __init__(
            self, 
            timesteps, 
            time_embedding_dim, 
            image_size, 
            in_channels=3, 
            out_channels=3, 
            base_dim=128, 
            dim_mults=[1,2,4,8]
        ):

        super().__init__()
        self.timesteps = timesteps
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        channels = self._cal_channels(base_dim, dim_mults)

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.encoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleDict()

        attention_resolutions = {16: 'attn_16'}
        current_resolution = image_size

        # Initialize the encoder blocks and attention blocks

        for _, (in_ch, out_ch) in enumerate(channels):

            self.encoder_blocks.append(EncoderBlock(in_ch, out_ch, time_embedding_dim))
            current_resolution //= 2

            if current_resolution in attention_resolutions:
                self.attention_blocks[attention_resolutions[current_resolution]] = AttnBlock(out_ch)

        self.mid_block = nn.Sequential(*[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
                                       ResidualBottleneck(channels[-1][1], channels[-1][1]))
        
        # Start with the lowest resolution achieved by the encoder
        current_resolution = image_size // (2 ** len(dim_mults))
        print(channels)
        reversed_channels = channels[::-1]

        # Initialize the decoder blocks
        self.decoder_blocks = nn.ModuleList()
        # Initialize the first decoder block, which does not have a skip connection yet
        self.decoder_blocks.append(DecoderBlock(reversed_channels[0][1], reversed_channels[1][0], time_embedding_dim))

        # Initialize the rest of the decoder blocks with skip connections
        for idx in range(1, len(reversed_channels)):

            in_ch = reversed_channels[idx - 1][0] + channels[idx - 1][1]
            out_ch = channels[idx][0] if idx < len(channels) - 1 else out_channels
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch, time_embedding_dim))

        self.final_conv = nn.Conv2d(channels[0][0], out_channels, kernel_size=1)


    def forward(self, x, t=None):
        x = self.init_conv(x)
        t = self.time_embedding(t) if t is not None else None

        encoder_shortcuts = []

        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)

            # print(x.size())
            attn_key = f'attn_{x.shape[-1]}'
            if attn_key in self.attention_blocks:
                x = self.attention_blocks[attn_key](x)

        x = self.mid_block(x)

        # Reverse the encoder shortcuts to match the upscaling order in the decoder
        encoder_shortcuts.reverse()
        
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):

            x = decoder_block(x, shortcut, t)
            print(x.size())

            # Apply attention after the corresponding decoder block
            attn_key = f'attn_{x.shape[-1]}'
            if attn_key in self.attention_blocks:
                x = self.attention_blocks[attn_key](x)

        x = self.final_conv(x)
        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims)-1):
            channels.append((dims[i], dims[i+1]))  # (in_channel, out_channel)
        return channels


class UnetWithAttn(nn.Module):
    def __init__(self, timesteps, time_embedding_dim, in_channels=3, out_channels=2, base_dim=32, dim_mults=(2, 4, 8, 16), dropout_rate=0.0):
        super().__init__()

        channels = self._cal_channels(base_dim, dim_mults)
        self.timesteps = timesteps

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        
        # Encoder blocks and attention blocks for skip connections
        self.encoder_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        prev_channels = base_dim

        for dim_mult in dim_mults:
            out_channels = base_dim * dim_mult
            self.encoder_blocks.append(EncoderBlock(prev_channels, out_channels, time_embedding_dim))
            self.attn_blocks.append(AttnBlock(prev_channels, dropout_rate))
            prev_channels = out_channels

        self.mid_block = nn.Sequential(
            ResidualBottleneck(prev_channels, prev_channels),
            ResidualBottleneck(prev_channels, prev_channels)
        )
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()

        prev_channels = prev_channels // 2

        for i in range(len(dim_mults) - 1, -1, -1):

            skip_channels = base_dim * dim_mults[i]

            in_channels = prev_channels + skip_channels
            out_channels = base_dim * dim_mults[i]
            
            print(prev_channels, skip_channels, out_channels)

            self.decoder_blocks.append(DecoderBlock(in_channels, out_channels, time_embedding_dim))
            # The output channels of this block will be the input channels for the next (earlier in the U-Net)
            prev_channels = out_channels


        self.final_conv = nn.Conv2d(in_channels=prev_channels // 2, out_channels=out_channels, kernel_size=1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        if t is not None:
            assert 0 <= t.min() and t.max() < self.timesteps, "Time step 't' is out of range."
            time_emb = self.time_embedding(t)

        # Encoder path with skip connections
        skip_connections = []

        for encoder_block, attn_block in zip(self.encoder_blocks, self.attn_blocks):

            x, x_shortcut = encoder_block(x, time_emb) 
            x_shortcut = attn_block(x_shortcut)  
            skip_connections.append(x_shortcut)  

        x = self.mid_block(x)

        for decoder_block, skip_connection in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder_block(x, skip_connection, time_emb)
            print(x.size())

        # Final convolution
        x = self.final_conv(x)

        return x


    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims)-1):
            channels.append((dims[i], dims[i+1]))  # (in_channel, out_channel)
        return channels

class Unet(nn.Module):
    '''
    simple unet design without attention
    '''
    def __init__(self,timesteps,time_embedding_dim,in_channels=3,out_channels=2,base_dim=32,dim_mults=[2,4,8,16]):
        super().__init__()
        assert isinstance(dim_mults,(list,tuple))
        assert base_dim%2==0 

        channels=self._cal_channels(base_dim,dim_mults)

        self.init_conv=ConvBnSiLu(in_channels,base_dim,3,1,1)
        self.time_embedding=nn.Embedding(timesteps,time_embedding_dim)

        self.encoder_blocks=nn.ModuleList([EncoderBlock(c[0],c[1],time_embedding_dim) for c in channels])
        self.decoder_blocks=nn.ModuleList([DecoderBlock(c[1],c[0],time_embedding_dim) for c in channels[::-1]])
    
        self.mid_block=nn.Sequential(*[ResidualBottleneck(channels[-1][1],channels[-1][1]) for i in range(2)],
                                        ResidualBottleneck(channels[-1][1],channels[-1][1]//2))

        self.final_conv=nn.Conv2d(in_channels=channels[0][0]//2,out_channels=out_channels,kernel_size=1)

    def forward(self,x,t=None):
        x=self.init_conv(x)
        if t is not None:
            t=self.time_embedding(t)
        encoder_shortcuts=[]
        for encoder_block in self.encoder_blocks:
            x,x_shortcut=encoder_block(x,t)
            encoder_shortcuts.append(x_shortcut)
        x=self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block,shortcut in zip(self.decoder_blocks,encoder_shortcuts):
            x=decoder_block(x,shortcut,t)
        x=self.final_conv(x)

        return x

    def _cal_channels(self,base_dim,dim_mults):
        dims=[base_dim*x for x in dim_mults]
        dims.insert(0,base_dim)
        channels=[]
        for i in range(len(dims)-1):
            channels.append((dims[i],dims[i+1])) # in_channel, out_channel

        return channels




if __name__=="__main__":
    x = torch.randn(3, 3, 224, 224)  # A batch of 3 images with 3 channels and 224x224 pixels
    t = torch.randint(0, 1000, (3,))  # Random time steps for each image in the batch
    model = UnetWithAttn(1000, 128, 3, 2, 32)  # Initialize your model with the correct parameters
    output = model(x, t)  # Run the forward pass with the images and time steps

    print(output.shape)
