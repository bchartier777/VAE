import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Encoder_v1(nn.Module):
    def __init__(self, input_dim, h_dim, latent_dim):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, h_dim)
        self.mean = nn.Linear(h_dim, latent_dim)
        self.var = nn.Linear(h_dim, latent_dim)

    def forward(self, x):
        h1 = F.relu(self.lin1(x))
        h2 = F.relu(self.lin2(h1))
        h3 = F.relu(self.lin3(h2))
        mean, var = self.mean(h3), self.var(h3)
        return mean, var
	
class Decoder_v1(nn.Module):
    def __init__(self, latent_dim, h_dim, input_dim):
        super().__init__()

        self.lin1 = nn.Linear(latent_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, input_dim)

	# To do: Add 'return torch.distributions.Normal' logic from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
    def forward(self, x):
        x1 = F.relu(self.lin1(x))
        img = torch.sigmoid(self.lin2(x1))
        return img

class Encoder_v2(nn.Module):
    def __init__(self, input_dim, h_dim, latent_dim):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, h_dim)
        self.mean = nn.Linear(h_dim, latent_dim)
        self.var = nn.Linear(h_dim, latent_dim)

    def forward(self, x):
        h1 = F.relu(self.lin1(x))
        mean, var = self.mean(h1), self.var(h1)
        return mean, var
	
class Decoder_v2(nn.Module):
    def __init__(self, latent_dim, h_dim, input_dim):
        super().__init__()

        self.lin1 = nn.Linear(latent_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, input_dim)

	# To do: Add 'return torch.distributions.Normal' logic from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
    def forward(self, x):
        x1 = F.relu(self.lin1(x))
        img = torch.sigmoid(self.lin2(x1))
        return img
