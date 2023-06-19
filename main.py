# This is an implementation of a standard variational autoencoder.
# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
# https://arxiv.org/abs/1312.6114

import torch
import torch.optim as optim
from model import VAE
from utils import parse_args, process_data_grayscale_v1, to_device
from train import train_model

def main(args):
    torch.manual_seed(args.seed)

    # Download and extract MNIST data
    train_data, val_data = process_data_grayscale_v1(args)

    # Initialize model and subset of variables
    model = VAE(args)
    model = to_device(model)
    name = model.__class__.__name__
    model.dim = int(args.x_size ** 0.5)
    optimizer = optim.AdamW(params = model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)

    # Train model and save validation images - not using the test dataset with this version
    train_model(model, args, train_data, val_data, optimizer, name)

if __name__ == '__main__':
	args = parse_args() # Parse arguments from command line
	main(args)
