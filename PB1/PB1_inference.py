import random
import sys, os
import argparse, torch, os
import torch
from torchvision.utils import save_image
from DDPM import DDPM 
from model import Unet


def main(out_dir):
    ckpt_dir = "./bestmodel_PB1.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # $ python PB1_inference.py /home/weihsin/project/dlcv-fall-2024-hw2-weihsinyeh/PB1_output
    model = DDPM( model     = Unet( input_channels = 3, num_features = 128, num_classes = 20),
                betas     = (1e-4, 0.02),
                n_T       = 500,
                device    = device,
                drop_prob = 0.1)
                
    model = model.to(device)
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
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    return parser.parse_args()
if __name__ == "__main__":
    torch.manual_seed(42)
    args = parser()
    main(args.out_dir)