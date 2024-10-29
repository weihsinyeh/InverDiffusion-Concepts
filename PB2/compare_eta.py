from torchvision import models, transforms
from PIL import Image
from torchvision.utils import save_image
form ../PB2_inference import output_img
def eta_compare(output_dir, image_dir):
    ori_dir = "../PB2_output/"
    img_dir = "../PB2_output/eta_compare/"

    imgs_grid = torch.empty(0, dtype=torch.float32)

    for eta in np.arange(0, 1.25, 0.25):
        output_img(img_num=10, eta=eta, image_dir=image_dir)
        output_img(input_noise, output_dir, UNet_pretrain):
        imgs = [f"{image_dir}{i:02d}.png" for i in range(10)]
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

    save_dir = os.path.join(output_dir, "visualize/")
    os.makedirs(save_dir, exist_ok=True)
    save_image(imgs_grid, save_dir + "compare_eta.png")

# Usage : 
# $ python3 ./PB2/compare_eta.py ./PB2_output/ ./PB2_output/
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_noise', type=str, help='path to the directory of predefined noises')
    parser.add_argument('image_dir', type=str, help='path to the directory of generated images')
    parser.add_argument('output_dir', type=str, help='path to the directory for your 10 generated')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    eta_compare(image_dir=args.image_dir, GroundTruth_dir=args.GroundTruth_dir, output_dir= args.output_dir )