import random
import sys, os
from pathlib import Path

import torch
from torchvision.utils import save_image

from PB1.DDPM import DDPM 
from PB1.model import Unet

random.seed(0)
torch.manual_seed(0)

out_dir = Path(sys.argv[1])
try:
    out_dir.mkdir(exist_ok=True, parents=True)
except:
    pass
ckpt_dir = Path('./P1_ckpt/100_ddpm.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# $ python PB1_inference.py /home/weihsin/project/dlcv-fall-2024-hw2-weihsinyeh/PB1_output
model = DDPM(   model=Unet(   in_channels=3,
                                n_features=128,
                                n_classes=20),
                betas=(1e-4, 0.02),
                n_T=500,
                device=device,
                drop_prob=0.1).to(device)
model.load_state_dict(torch.load(ckpt_dir, map_location=device))
model.eval()
for class_idx in range(20):
    gened_count = 0
    with torch.no_grad():
        x_i, _ = model.class_gen(50, size=(3, 28, 28), device=device, class_idx=class_idx, guide_w=2)
    for image in x_i:
        gened_count += 1
        if (class_idx < 10):
            path = os.path.join(out_dir, 'mnistm')
            path = os.path.join(path, f'{class_idx}_{gened_count}.png')
            save_image(image, path)
        else:
            class_idx_svhn = class_idx - 10
            path = os.path.join(out_dir, 'svhn')
            path = os.path.join(path, f'{class_idx_svhn}_{gened_count}.png')
            save_image(image, path)
