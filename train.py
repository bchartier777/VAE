import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import display

from utils import to_device

def train_model(model, args, train_data, val_data, optimizer, name):
    agg_kl = []  # Aggregated LK Divergence loss
    recon_loss_agg = []  # Aggregated recon loss

    # Train model over user-defined epochs
    for epoch in tqdm(range(1, args.user_epochs+1)):

        # Set model to train mode to update weights
        model.train()
        ep_loss, agg_recon, agg_kld = [], [], []

        for batch in train_data:

            # Zero gradients and calculate losses over ELBO
            optimizer.zero_grad()
            loss_recon, kl_div = calc_loss_KLD(model, batch)
            agg_loss = loss_recon + kl_div

            # Update model params
            agg_loss.backward()
            optimizer.step()

            # Log losses
            ep_loss.append(agg_loss.item())
            agg_recon.append(loss_recon.item())
            agg_kld.append(kl_div.item())

        # Log aggregated losses
        agg_kl.extend(agg_kld)
        recon_loss_agg.extend(agg_recon)

        # Evaluate the model on the validation set - set to Eval mode to disable dropout
        # layers, batch norm, etc 
        model.eval()
        val_loss = calc_loss_bat(model, val_data)

        # Stop if it achieves acceptable loss on validation dataset
        if val_loss < args.lowest_val_loss:
            best_model = deepcopy(model)
            args.lowest_val_loss = val_loss

        # Output progress
        print ("epoch [%d/%d], Agg loss: %.4f, recon loss: %.4f, KLD: %.7f, Valid loss: %.4f"
               %(epoch, args.user_epochs, np.mean(ep_loss),
               np.mean(agg_recon), np.mean(agg_kld), val_loss))
        args.user_epochs += 1

        # Debugging and visualization purposes
        if args.val_output:
            save_sample(model, name, args, epoch)
            plt.show()
    print ("Training complete: epoch[%d/%d], Agg loss: %.4f, recon loss: %.4f, KLD: %.7f, Valid loss: %.4f"
           %(epoch, args.user_epochs, np.mean(ep_loss),
           np.mean(agg_recon), np.mean(agg_kld), val_loss))

# Calculate reconstruction and KLD loss for single image
def calc_loss_KLD(model, bat):
    img, _ = bat
    img = to_device(img.view(img.shape[0], -1))

    # Generate representation from encoder/decoder and calculate loss
    output, mean, var = model(img)
    loss_recon = torch.sum((img - output) ** 2)

    # KL div between encoder output and Normal dist
    KLD = torch.sum(0.5 * (mean**2 + torch.exp(var) - var - 1))

    return loss_recon, KLD

# Calculate reconstruction and Binary Cross-entropy loss for single image
def calc_loss_BCE(model, bat):
    img, _ = bat
    img = to_device(img.view(img.shape[0], -1))

    # Generate representation from encoder and calculate loss
    output, mean, var = model(img)

    BCE = F.binary_cross_entropy(output, img, size_average=False)

    KLD = torch.sum(0.5 * (mean**2 + torch.exp(var) - var - 1))
    return BCE, KLD

# Generate an image from the decoder and save to user-defined location
def save_sample(model, model_name, args, epoch):
    # Generate images by sampling z ~ p(z), x ~ p(x|z,θ)
    z = to_device(torch.randn(args.val_imgs, args.latent_dim))
    x = model.decoder(z)

    # Generate grid and save to output folder
    grid = make_grid(x.data.view(args.val_imgs,
                                            -1,
                                            model.dim,
                                            model.dim),
                     nrow=int(args.val_imgs**0.5))

    folder = 'image_val/'
    file_name = 'sample_%d.png' %(epoch)
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_image(grid, folder + file_name)

# Calculate reconstruction and KLD loss for a batch of images
def calc_loss_bat(model, dataset):
    loss_bat = []
    for img in dataset:
        # loss_recon, kld = calc_loss_BCE(model, img)
        loss_recon, kld = calc_loss_KLD(model, img)
        agg_loss = loss_recon + kld
        loss_bat.append(agg_loss.item())

    loss_bat = np.mean(loss_bat)
    return loss_bat
