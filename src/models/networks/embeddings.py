import math

import torch
from torch import nn
import torch.nn.functional as F


def get_1d_sincos_encode(steps: torch.Tensor, emb_dim: int, max_period: int=10000) -> torch.Tensor:
    """Get sinusoidal encodings for a batch of timesteps/positions."""
    assert steps.dim() == 1, f"Parameter `steps` must be a 1D tensor, but got {steps.dim()}D."

    half_dim = emb_dim // 2
    emb = torch.exp(- math.log(max_period) *\
        torch.arange(0, half_dim, device=steps.device).float() / half_dim)
    emb = steps[:, None].float() * emb[None, :]  # (num_steps, half_dim)

    # Concat sine and cosine encodings
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # (num_steps, emb_dim)

    # Zero padding
    if emb_dim % 2 == 1: emb = nn.functional.pad(emb, (0, 1))
    assert emb.shape == (steps.shape[0], emb_dim)

    return emb

def get_3d_positional_encode(positions: torch.Tensor, emb_dim: int, max_period: int=10000):
    """Get 3d sinusoidal encodings for a batch of 3D positions.
        positions: (b, n, 3)
        emb_dim: 512
    """
    one_position_dim = emb_dim//3//2*2
    positions = positions*(max_period/2)+max_period
    emb_x = positions[:,:,0] #(b, n)
    emb_y = positions[:,:,1]
    emb_z = positions[:,:,2]
    dim = torch.arange(one_position_dim, device=positions.device) 
    dim = max_period**(2*(dim//2)/one_position_dim) # (d//3,)
    pos_x = emb_x[...,None]/dim                     # (b, n, d//3)
    pos_x = torch.stack((pos_x[:, : , 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2) 
    pos_y = emb_y[...,None]/dim                     # (b, n, d//3)
    pos_y = torch.stack((pos_y[:, : , 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2) 
    pos_z = emb_z[...,None]/dim                     # (b, n, d//3)
    pos_z = torch.stack((pos_z[:, : , 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2) 
    pos_encode = torch.cat((pos_x, pos_y, pos_z), dim=-1) #(b, n, d)
    pos_encode = F.pad(pos_encode, (0, emb_dim-pos_encode.shape[-1]), 'constant', 0) #(b, n, d)
    return pos_encode
    
    


class Timestep(nn.Module):
    """Encode timesteps with sinusoidal encodings."""
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_1d_sincos_encode(timesteps, self.time_emb_dim)


class TimestepEmbed(nn.Module):
    """Embed sinusoidal encodings with a 2-layer MLP."""
    def __init__(self, in_dim: int, time_emb_dim: int, act_fn_name: str="GELU"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, time_emb_dim),
            getattr(nn, act_fn_name)(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.mlp(sample)
