from PB2.DDIM import DDIM
from PB2.UNet import UNet
from PB2.utils import beta_scheduler
from torchvision.utils import save_image
import argparse, torch, os
def output_img( input_noise, 
                output_dir, 
                UNet_pretrain):
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set the output directory for the generated images
    save_dir = output_dir

    # load the pretrained UNet model
    UNet_pt_dir = UNet_pretrain
    print(f"Loading the pretrained UNet model from {UNet_pt_dir}")
    unet_model = UNet()

    unet_model.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM( model=unet_model.to(device), timesteps=n_T, beta_schedule=beta_scheduler())

    with torch.no_grad():
        
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

'''
$1: path to the directory of predefined noises (e.g. “~/hw2/DDIM/input_noise”)
$2: path to the directory for your 10 generated images (e.g. “~/hw2/DDIM/output_images”)
$3: path to the pretrained model weight(e.g. “~/hw2/DDIM/UNet.pt”
Usage:
$ python3 .PB2/PB2_inference.py ./hw2_data/DDIM/input_noise/ ./hw2_data/DDIM/output_images/ ./hw2_data/DDIM/UNet.pt
'''
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_noise', type=str, help='path to the directory of predefined noises')
    parser.add_argument('output_images', type=str, help='path to the directory for your 10 generated images')
    parser.add_argument('UNet_pt', type=str, help='path to the pretrained model weight')
    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)
    args = parser()
    output_img(args.input_noise, args.output_images, args.UNet_pt)