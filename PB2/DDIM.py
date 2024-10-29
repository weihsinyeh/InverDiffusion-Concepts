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
from utils import beta_scheduler

class DDIM:
    def __init__(self, model, timesteps=1000, beta_schedule=beta_scheduler()):
        self.model = model
        self.timesteps = timesteps
        self.betas = beta_schedule

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def slerp(self, t, v0, v1):
        """Spherical Linear Interpolation (Slerp) between v0 and v1 by ratio t."""
        dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)), dim=1, keepdim=True)
        theta = torch.acos(dot) * t
        sin_theta = torch.sin(theta)
        sin_theta_0 = torch.sin(torch.acos(dot))
        s0 = torch.sin((1 - t) * theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * v0 + s1 * v1

    # use ddim to sample
    @torch.no_grad()
    def sample( self, predefined_noises, batch_size=10, ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True):
        c = self.timesteps // ddim_timesteps
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c))) + 1
        print(ddim_timestep_seq) # check

        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.model.parameters()).device
        # start from pure noise (for each example in the batch)
        filenames = [f"{i:02d}.pt" for i in range(0, batch_size)]

        tensors = [
            torch.load(os.path.join(predefined_noises, filename))
            for filename in filenames
        ]
        sample_img = torch.cat(tensors, dim=0)

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc="Sampling loop time step", total=ddim_timesteps,):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long,)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            pred_noise = self.model(sample_img, t)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = (torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise)

            # 6. compute x_{t-1} of formula (12)
            x_prev = (torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img))
            sample_img = x_prev

        return sample_img.cpu()

    def slerp_sample(self, predefined_noises, batch_size=10, ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True):
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c))) + 1
        # add one to get the final alpha values right (the ones from first scale to data during sampling)

        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.model.parameters()).device
        filenames = [f"{i:02d}.pt" for i in range(batch_size)]

        tensors = [torch.load(os.path.join(predefined_noises, filename)) for filename in filenames]
        sample_img = torch.cat(tensors, dim=0)

        for i in tqdm(reversed(range(ddim_timesteps)), desc="Sampling loop time step", total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            pred_noise = self.model(sample_img, t)
            pred_x0 = (sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

            # Apply Slerp for interpolation
            noise = torch.randn_like(sample_img)
            sample_img = self.slerp(ddim_eta, sample_img, noise)

            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * noise
            sample_img = x_prev

        return sample_img.cpu()
    

def eta_compare():
    ori_dir = "../PB2_output/"
    img_dir = "../PB2_output/eta_compare/"

    imgs_grid = torch.empty(0, dtype=torch.float32)

    for eta in np.arange(0, 1.25, 0.25):
        output_img(img_num=10, eta=eta, image_dir=img_dir, interpolation = linear)
        imgs = [f"{img_dir}{i:02d}.png" for i in range(10)]
        imgs_row = torch.empty(0, dtype=torch.float32)
        for img in imgs:
            img = transforms.ToTensor()(Image.open(img))
            imgs_row = torch.cat((imgs_row, img), dim=2)

        blank_column = torch.ones(3, 256, 256) * 255
        imgs_row = torch.cat((blank_column, imgs_row), dim=2)
        pil_image = transforms.ToPILImage()(imgs_row)  # Tensor to PIL
        draw = ImageDraw.Draw(pil_image)
        draw.text(
            xy=(blank_column.shape[0] // 3, blank_column.shape[1] // 2),
            text=f"eta = {eta}: ",
            fill=(255, 255, 255),
        )
        imgs_row = transforms.ToTensor()(pil_image)
        imgs_grid = torch.cat((imgs_grid, imgs_row), dim=1)

    save_image(imgs_grid, img_dir + "grid.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    output_img()
    eta_compare()
    Compare_mse()
