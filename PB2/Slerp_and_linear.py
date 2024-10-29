import os, sys, torch
import numpy as np
from PIL import Image
from UNet import UNet
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from DDIM import DDIM
from utils import beta_scheduler
def slerp(x1, x2, alpha):
    theta = torch.acos(torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2)))
    x_interpolation = torch.sin((1 - alpha) * theta) / torch.sin(theta) * x1 + torch.sin(alpha * theta) / torch.sin(theta) * x2
    return x_interpolation

def linear(x1, x2, alpha):
    x_interpolation = (1 - alpha) * x1 + alpha * x2
    return x_interpolation

def output_img(batch=11, eta=0):
    # hardcoding these here
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_path  = "./hw2_data/face/noise"
    save_dir    = "./PB2_output_test/vis"
    UNet_pt_dir = "./hw2_data/face/UNet.pt"
    unet_model  = UNet()
    unet_model.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM( model=unet_model.to(device), timesteps=n_T, beta_schedule = beta_scheduler())

    with torch.no_grad():
        x_gen = ddim.sample(predefined_noises=noise_path, batch_size=batch, ddim_eta=eta, interpolation=slerp)
        concat = []
        for i in range(len(x_gen)):
            img = x_gen[i]
            min_val, max_val = torch.min(img), torch.max(img)

            # Min-Max Normalization
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            concat.append(normalized_x_gen)
            save_image(normalized_x_gen, save_dir + f"/{i:02d}.png")

        concat = torch.cat(concat, dim=2)
        save_image(concat, save_dir + f"/concat_slerp.png")

        x_gen = ddim.sample(predefined_noises=noise_path, batch_size=batch, ddim_eta=eta, interpolation=linear)
        concat = []
        for i in range(len(x_gen)):
            img = x_gen[i]
            min_val, max_val = torch.min(img), torch.max(img)

            # apply min-max normalization to the DDIM output images 
            # before saving them with torchvision.utils.save_image, to avoid contrast issues.
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            concat.append(normalized_x_gen)
            save_image(normalized_x_gen, save_dir + f"/{i:02d}.png")

        concat = torch.cat(concat, dim=2)
        save_image(concat, save_dir + f"/concat_linear.png")


if __name__ == "__main__":
    output_img()