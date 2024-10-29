from torchvision import models, transforms
from PIL import Image
from torchvision.utils import save_image
from DDIM import DDIM
from UNet import UNet
from torchvision.utils import save_image
import argparse, torch, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
def output_img( input_noise, 
                output_dir, 
                UNet_pretrain, 
                interpolation = "linear"):
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set the output directory for the generated images
    save_dir = output_dir

    # load the pretrained UNet model
    UNet_pt_dir = UNet_pretrain
    print(f"Loading the pretrained UNet model from {UNet_pt_dir}")
    unet_model = UNet()

    unet_model.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM( model=unet_model.to(device), timesteps=n_T)

    with torch.no_grad():
        if interpolation == "linear":
            x_gen = ddim.sample(input_noise,batch_size=10, ddim_eta=0)
            for i in range(len(x_gen)):
                img = x_gen[i]
                min_val = torch.min(img)
                max_val = torch.max(img)

                # apply min-max normalization to the DDIM output images 
                # before saving them with torchvision.utils.save_image, to avoid contrast issues.
                normalized_x_gen = (img - min_val) / (max_val - min_val)
                save_path = os.path.join(save_dir, f"{i:02d}.png")

                save_image(normalized_x_gen, save_path)
        else:
            x_gen = ddim.slerp_sample(input_noise,batch_size=10, ddim_eta=0)
            for i in range(len(x_gen)):
                img = x_gen[i]
                min_val = torch.min(img)
                max_val = torch.max(img)

                # apply min-max normalization to the DDIM output images 
                # before saving them with torchvision.utils.save_image, to avoid contrast issues.
                normalized_x_gen = (img - min_val) / (max_val - min_val)
                save_path = os.path.join(save_dir, f"{i:02d}.png")

                save_image(normalized_x_gen, save_path)

def eta_compare(input_noise,
                image_dir, 
                output_dir, 
                UNet_pretrain):

    imgs_grid = torch.empty(0, dtype=torch.float32)

    for eta in np.arange(0, 1.25, 0.25):
        # Slerp interpolation
        output_dir_slerp = os.path.join(output_dir, "slerp/")
        os.makedirs(output_dir_slerp, exist_ok=True)
        output_img(input_noise, output_dir_slerp, UNet_pretrain, interpolation="slerp")

        # Linear interpolation
        output_dir_linear = os.path.join(output_dir, "linear/")
        os.makedirs(output_dir_linear, exist_ok=True)
        output_img(input_noise, output_dir_linear, UNet_pretrain, interpolation="linear")

        # Collect images for the current eta
        imgs_row = torch.empty(0, dtype=torch.float32)

        # Combine images for slerp
        imgs_slerp = [f"{output_dir_slerp}{i:02d}.png" for i in range(10)]
        for img in imgs_slerp:
            img_tensor = transforms.ToTensor()(Image.open(img))
            imgs_row = torch.cat((imgs_row, img_tensor), dim=2)

        # Add blank column and label for slerp
        blank_column = torch.ones(3, 256, 256) * 255
        imgs_row = torch.cat((blank_column, imgs_row), dim=2)
        pil_image = transforms.ToPILImage()(imgs_row)
        draw = ImageDraw.Draw(pil_image)
        draw.text(
            xy=(blank_column.shape[1] // 3, blank_column.shape[0] // 2),
            text=f"Slerp eta = {eta}: ",
            fill=(255, 255, 255),
        )
        imgs_row_slerp = transforms.ToTensor()(pil_image)

        # Combine images for linear
        imgs_row_linear = torch.empty(0, dtype=torch.float32)
        imgs_linear = [f"{output_dir_linear}{i:02d}.png" for i in range(10)]
        for img in imgs_linear:
            img_tensor = transforms.ToTensor()(Image.open(img))
            imgs_row_linear = torch.cat((imgs_row_linear, img_tensor), dim=2)

        # Add blank column and label for linear
        imgs_row_linear = torch.cat((blank_column, imgs_row_linear), dim=2)
        pil_image = transforms.ToPILImage()(imgs_row_linear)
        draw = ImageDraw.Draw(pil_image)
        draw.text(
            xy=(blank_column.shape[1] // 3, blank_column.shape[0] // 2),
            text=f"Linear eta = {eta}: ",
            fill=(255, 255, 255),
        )
        imgs_row_linear = transforms.ToTensor()(pil_image)

        # Concatenate Slerp and Linear rows to the grid
        imgs_grid = torch.cat((imgs_grid, imgs_row_slerp, imgs_row_linear), dim=1)

    # Save final grid image
    save_image(imgs_grid, os.path.join(output_dir, "eta_comparison_grid.png"))
    
    

    save_dir = os.path.join(output_dir, "visualize/")
    os.makedirs(save_dir, exist_ok=True)
    save_image(imgs_grid, save_dir + "compare_eta.png")

# Usage : 
# $ python3 ./PB2/compare_eta.py ./hw2_data/face/noise/ ./PB2_output/eta_compare/ ./PB2_output/eta_compare/ ./hw2_data/face/UNet.pt

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_noise', type=str, help='path to the directory of predefined noises')
    parser.add_argument('image_dir', type=str, help='path to the directory of generated images')
    parser.add_argument('output_dir', type=str, help='path to the directory for your 10 generated')
    parser.add_argument('UNet_pt', type=str, help='path to the pretrained model weight')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    eta_compare(input_noise=args.input_noise,
                image_dir=args.image_dir, 
                output_dir=args.output_dir, 
                UNet_pretrain= args.UNet_pt )