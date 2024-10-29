from PB2.DDIM import DDIM
from PB2.UNet import UNet
from torchvision.utils import save_image
import argparse
def output_img(args):
    # hardcoding these here
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = args.output_images
    UNet_pt_dir = args.UNet_pt
    unet_model = UNet()
    unet_model.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM(
        model=unet_model.to(device),
        timesteps=n_T,
    )

    with torch.no_grad():
        x_gen = ddim.sample(args.input_noise,batch_size=10, ddim_eta=0)
        for i in range(len(x_gen)):
            img = x_gen[i]
            min_val = torch.min(img)
            max_val = torch.max(img)

            # Min-Max Normalization
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            save_image(normalized_x_gen, save_dir + f"{i:02d}.png")

'''
 $1: path to the directory of predefined noises (e.g. “~/hw2/DDIM/input_noise”)
 $2: path to the directory for your 10 generated images (e.g. “~/hw2/DDIM/output_images”)
 $3: path to the pretrained model weight(e.g. “~/hw2/DDIM/UNet.pt”
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
    output_img(args)