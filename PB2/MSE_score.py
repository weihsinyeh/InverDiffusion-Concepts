import argparse, torch, os
from torchvision import models, transforms
from PIL import Image
from torchvision.utils import save_image

def normalalize_to_255(tensor):
    for i, x in enumerate(tensor):
        new_values = []
        for row in x:
            new_row = []
            for value in row:
                new_value = int(value * 255)
                new_row.append(new_value)
            new_values.append(new_row)
        tensor[i] = torch.tensor(new_values)
    return tensor

def Compare_mse(image_dir, GroundTruth_dir, output_dir):
    image       = [f"{image_dir}{i:02d}.png" for i in range(10)]
    groud_truth = [f"{GroundTruth_dir}{i:02d}.png" for i in range(10)]
    transform   = transforms.Compose([transforms.ToTensor(),])

    compare_image_list = torch.empty(0, dtype=torch.float32)
    for i, (generated_path, ground_truth_path) in enumerate(zip(image, groud_truth)):
        image       = transform(Image.open(generated_path))
        groud_truth = transform(Image.open(ground_truth_path))
        
        compare_image       = torch.cat((image, groud_truth), dim=1)
        compare_image_list  = torch.cat((compare_image_list, compare_image), dim=2)

        image       = normalalize_to_255(image)
        groud_truth = normalalize_to_255(groud_truth)

        mse = torch.nn.functional.mse_loss(image, groud_truth)
        print(f"MSE for image pair {i}: {mse.item():5f}")

    # Save the compare image
    save_image(compare_image_list, "compareMSE.png")

# Usage : 
# $ python3 ./PB2/MSE_score.py ./PB2_output/ ./hw2_data/face/GT/ ./PB2_output/
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='path to the directory of generated images')
    parser.add_argument('GroundTruth_dir', type=str, help='path to the directory of ground truth images')
    parser.add_argument('output_dir', type=str, help='path to the directory for your 10 generated')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    Compare_mse(image_dir=args.image_dir, GroundTruth_dir=args.GroundTruth_dir, output_dir= args.output_dir )