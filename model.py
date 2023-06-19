# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
# https://arxiv.org/abs/1312.6114

import torch
import torch.nn as nn

from utils import *
from network import *


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.encoder = Encoder_v1(input_dim=args.x_size, h_dim=args.h_dim, latent_dim=args.latent_dim)
        self.decoder = Decoder_v1(latent_dim=args.latent_dim, h_dim=args.h_dim, input_dim=args.x_size)

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparam(mean, var)
        img = self.decoder(z)
        return img, mean, var
    
    """ Reparameterization trick. Instead of sampling from Q(z|X), 
    sample eps = N(z_mean,I) z = z_mean + sqrt(var)*eps.   """
    def reparam(self, z_mean, var):
        eps = to_device(torch.randn(z_mean.shape))
        z = z_mean + torch.exp(var/2) * eps
        return z


