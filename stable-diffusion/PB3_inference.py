import torch, os, random, itertools, PIL, json, argparse
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.util import instantiate_from_config
from transformers import CLIPTokenizer
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import numpy as np
from torch import autocast
from einops import rearrange
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    # Check if images is a PyTorch tensor and convert it to NumPy
    if isinstance(images, torch.Tensor):
        # Move to CPU and convert to NumPy
        images = images.cpu().numpy()

    if images.ndim == 3:
        images = images[None, ...]

    # Scale and clamp the images to [0, 255]
    images = (images * 255).round().clip(0, 255).astype(np.uint8)
    # Convert each image in the batch to a PIL Image
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def main(input_json_path, output_image_dir, checkpoint_path):
    config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    new1_checkpoint_path = "/project/g/r13922043/hw2/checkpoints/1030_0/fine_tuned_4.ckpt"
    new2_checkpoint_path = "/project/g/r13922043/hw2/checkpoints/1030_1/fine_tuned_David1.ckpt"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load Model Configuration and Checkpoint
    config  = OmegaConf.load(config_path)
    model   = load_model_from_config(config, checkpoint_path)
    model.eval()
    sampler = DPMSolverSampler(model)

    # Initialize CLIPTokenizer and add placeholder token
    tokenizer = model.cond_stage_model.tokenizer

    new1_placeholder_token = "<new1>"
    new2_placeholder_token = "<new2>"
    tokenizer.add_tokens(new1_placeholder_token)
    tokenizer.add_tokens(new2_placeholder_token)
    new1_learned_embeds_dict = torch.load(new1_checkpoint_path, map_location=device)[new1_placeholder_token]
    new2_learned_embeds_dict = torch.load(new2_checkpoint_path, map_location=device)[new2_placeholder_token]
    new1_token_id = tokenizer.convert_tokens_to_ids(new1_placeholder_token)
    new2_token_id = tokenizer.convert_tokens_to_ids(new2_placeholder_token)
    text_encoder  = model.cond_stage_model
    text_encoder.transformer.resize_token_embeddings(len(tokenizer))
    text_encoder.transformer.get_input_embeddings().weight.data[new1_token_id] = new1_learned_embeds_dict
    text_encoder.transformer.resize_token_embeddings(len(tokenizer))
    text_encoder.transformer.get_input_embeddings().weight.data[new2_token_id] = new2_learned_embeds_dict

    with open(input_json_path, 'r') as f:
        prompt_data = json.load(f)

    for source_num, details in prompt_data.items():
        combined_prompts    = details["prompt"] + details["prompt_4_clip_eval"]

        sorce_output_dir = os.path.join(output_image_dir, f"{source_num}")
        os.makedirs(sorce_output_dir, exist_ok = True)
        # Iterate over each prompt for this source
        for prompt_num, prompt_text in enumerate(combined_prompts):
            prompt_output_dir = os.path.join(sorce_output_dir, f"{prompt_num}")
            os.makedirs(prompt_output_dir, exist_ok=True)

            # Generate image
            with autocast("cuda"):
                base_count = 0
                for i in range(5):
                    # Set up the conditioning and image generation
                    conditioning    = model.get_learned_conditioning([prompt_text] * 5)
                    shape = [4, 512 // 8, 512 // 8]
                    uc = model.get_learned_conditioning(5 * [""])
                    samples_ddim, _ = sampler.sample(   S=50,
                                                        conditioning=conditioning,
                                                        batch_size=5,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=7.5,
                                                        unconditional_conditioning=uc,
                                                        eta=0.0,
                                                        x_T=None)
                    x_samples_ddim  = model.decode_first_stage(samples_ddim)
                    x_samples_ddim  = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    for x_sample in x_checked_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(prompt_output_dir, f"source{source_num}_prompt{prompt_num}_{base_count}.png"))
                        base_count += 1
        

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json_path',  type=str, help='path to the directory of predefined noises')
    parser.add_argument('output_image_dir', type=str, help='path to your output folder (e.g. “~/output_folder”)')
    parser.add_argument('checkpoint_path',  type=str, help='path to the pretrained model weight (e.g. “~/hw2/personalization/model.ckpt”)')
    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)
    args = parser()
    os.makedirs(args.output_image_dir, exist_ok= True)
    main(args.input_json_path, args.output_image_dir, args.checkpoint_path)