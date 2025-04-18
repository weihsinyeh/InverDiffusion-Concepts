import random
import sys, torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
from PB1.DDPM import DDPM 
from PB1.model import Unet

random.seed(0)
torch.manual_seed(0)

ckpt_dir = Path('../P1_ckpt/100_ddpm.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DDPM(   model=Unet(in_channels=3, n_features=128, n_classes=20),
                betas=(1e-4, 0.02),
                n_T=500,
                device=device,
                drop_prob=0.1)

model = model.to(device)
model.load_state_dict(torch.load(ckpt_dir, map_location=device))
model.eval()
mnist = True
with torch.no_grad():
    x_i, x_i_store = model.sample(100, size=(3, 28, 28), device=device, guide_w=2)
    x_i = x_i.reshape(10, 10, 3, 28, 28)
    x_i = torch.transpose(x_i, 0, 1)
    x_i = x_i.reshape(-1, 3, 28, 28)
    save_image(x_i, '../PB1_plot/100_samples.png', nrow=10)
    if mnist == False:
        for i in range(10):
            save_image(torch.tensor(x_i_store[:, i, ...].reshape(32, 3, 28, 28)), f'../PB1_plot/svhn/{i}_sample_progress.png')
    else:
        for i in range(10):
            save_image(torch.tensor(x_i_store[:, i, ...].reshape(32, 3, 28, 28)), f'../PB1_plot/mnist/{i}_sample_progress.png')
