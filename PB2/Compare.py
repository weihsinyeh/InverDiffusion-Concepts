from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms as trns
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from DDIM import DDIM
from UNet import UNet
from utils import beta_scheduler
def output_img(img_num=10, eta=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    UNet_pt_dir = "../hw2_data/face/UNet.pt"
    input_noise = "../hw2_data/face/noise/"
    unetmodel = UNet()
    unetmodel.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM( model=unetmodel.to(device), timesteps=1000, beta_schedule = beta_scheduler())
    image_list = []
    with torch.no_grad():
        x_gen = ddim.sample(input_noise,batch_size=10, ddim_eta=eta)
        for i in range(len(x_gen)):
            img = x_gen[i]
            min_val, max_val = torch.min(img),  torch.max(img)

            # apply min-max normalization to the DDIM output images 
            # before saving them with torchvision.utils.save_image, to avoid contrast issues.
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            image_list.append(normalized_x_gen)
            save_image(normalized_x_gen, "../PB2_eta/" + f"{i:02d}.png")

def eta_compare():
    imgs_grid = torch.empty(0, dtype=torch.float32)

    for eta in np.arange(0, 1.25, 0.25):
        output_img(img_num=10,eta=eta)
        image_list = [f"../PB2_eta/{i:02d}.png" for i in range(10)]
        imgs_row = torch.empty(0, dtype=torch.float32)
        for img in image_list:
            img = transforms.ToTensor()(Image.open(img))
            imgs_row    = torch.cat((imgs_row, img), dim=2)

        blank_column    = torch.ones(3, 256, 256) * 255
        imgs_row        = torch.cat((blank_column, imgs_row), dim=2)
        pil_image       = transforms.ToPILImage()(imgs_row)  # Tensor to PIL
        draw = ImageDraw.Draw(pil_image)
        draw.text(  xy=(blank_column.shape[0] // 3, blank_column.shape[1] // 2),
                    text=f"eta = {eta}: ",
                    fill=(255, 255, 255),)
        imgs_row        = transforms.ToTensor()(pil_image)
        imgs_grid       = torch.cat((imgs_grid, imgs_row), dim=1)

    save_image(imgs_grid, "grid.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    save_dir = "../PB2_eta/"
    os.makedirs(save_dir, exist_ok=True)
    output_img()
    eta_compare()
