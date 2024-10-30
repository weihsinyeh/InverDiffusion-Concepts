import torch, os, random, itertools, PIL, json, argparse
from tqdm import tqdm
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

def save_progress(text_encoder, placeholder_token_ids, placeholder_token, save_path, safe_serialization=True):
    learned_embeds = text_encoder.transformer.get_input_embeddings().weight[
        min(placeholder_token_ids) : max(placeholder_token_ids) + 1
    ]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

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

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.augmentation = transforms.RandomChoice([transforms.RandomApply([transforms.RandomHorizontalFlip()], p = 0.16)])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = text
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = ( img.shape[0], img.shape[1])
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.augmentation(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

def main(token_number):
    config_path = "./configs/stable-diffusion/v1-inference.yaml"
    checkpoint_path = "./ldm/models/stable-diffusion-v1/model.ckpt"

    output_image_dir = "/project/g/r13922043/hw2/output/1030_1"
    os.makedirs(output_image_dir, exist_ok=True)

    if token_number == 0 :
        image_folder = "/project/g/r13922043/hw2_data/textual_inversion/0"
        output_dir = "/project/g/r13922043/hw2/checkpoints/1030_0"
        os.makedirs(output_dir, exist_ok=True)
        placeholder_token = "<new1>"
        initializer_token = "Corgi"
        learnable_property = "object"
        input_json_path = "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input_0.json"
        # Load JSON file for evaluation
        with open(input_json_path, 'r') as f:
            prompt_data = json.load(f)
    elif token_number == 1 :
        image_folder = "/project/g/r13922043/hw2_data/textual_inversion/1"
        output_dir = "/project/g/r13922043/hw2/checkpoints/1030_1"
        os.makedirs(output_dir, exist_ok=True)
        placeholder_token = "<new2>"
        initializer_token = "cartoon"
        learnable_property = "style"
        input_json_path = "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input_1.json"
        # Load JSON file for evaluation
        with open(input_json_path, 'r') as f:
            prompt_data = json.load(f)
    else :
        image_folder = "/project/g/r13922043/hw2_data/textual_inversion/cat"
        output_dir = "/project/g/r13922043/hw2/checkpoints/1030_cat"
        os.makedirs(output_dir, exist_ok=True)
        placeholder_token = "<new3>"
        initializer_token = "cat"
        learnable_property = "object"


    # Load Model Configuration and Checkpoint
    config  = OmegaConf.load(config_path)
    model   = load_model_from_config(config, checkpoint_path)

    # Initialize CLIPTokenizer and add placeholder token
    tokenizer = model.cond_stage_model.tokenizer

    # Add new token
    num_added_tokens = tokenizer.add_tokens([placeholder_token])
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {placeholder_token}. Please use a unique token.")

    # Load Dataset
    train_data = TextualInversionDataset(data_root=image_folder,  tokenizer=tokenizer, placeholder_token=placeholder_token)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers= 4)

    # Convert token to IDs
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Load text encoder
    text_encoder = model.cond_stage_model
    text_encoder.transformer.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.transformer.get_input_embeddings().weight.data
    with torch.no_grad():
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

    optimizer = optim.AdamW(    text_encoder.transformer.get_input_embeddings().parameters(), lr=5e-3)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Training loop
    for epoch in range(50):  # Adjust epochs as needed
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} : Training Progress")):
            optimizer.zero_grad()

            batch["pixel_values"] = batch["pixel_values"].to(device)
            
            # Encode image to latent space
            latents = model.encode_first_stage(batch["pixel_values"])
            # Sample from the distribution to get the latent representation
            latents = latents.sample()
            latents = latents * 0.18215

            # Generate noise and timesteps
            noise = torch.randn(latents.shape).to(latents.device)
            timesteps = torch.randint(0, model.num_timesteps, (latents.shape[0],), device=latents.device).long()
            
            noisy_latents = model.q_sample(latents, timesteps, noise)

            token_id = tokenizer(batch["input_ids"], return_tensors="pt", padding=True, truncation=True)
            encoder_hidden_states = model.cond_stage_model(batch["input_ids"])
    
            # Forward pass for noise prediction
            noise_pred = model.apply_model(noisy_latents, timesteps, encoder_hidden_states)

            # Calculate and backpropagate the loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            loss.backward()
            # Only update new token gradients
            model_embeddings = text_encoder.transformer.get_input_embeddings().weight.grad
            # Check if gradients exist
            if model_embeddings is not None:
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                model_embeddings.data[index_grads_to_zero, :] = model_embeddings.data[index_grads_to_zero, :].fill_(0)

            # Perform optimizer step
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        '''
        model.cond_stage_model.tokenizer = tokenizer
        ckpt_name = "fine_tuned_" + str(epoch)+ ".ckpt"
        new_ckpt_path = os.path.join(output_dir, ckpt_name)
        checkpoint = {"state_dict": model.state_dict()}
        checkpoint["tokenizer"] = tokenizer
        # Save the model's state_dict inside the checkpoint dictionary
        torch.save(checkpoint, new_ckpt_path)
        '''
        
        # Save the newly trained embeddings
        # weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        # save_path = os.path.join(args.output_dir, weight_name)
        if token_number == 0:
            ckpt_name = "fine_tuned_" + str(epoch)+ ".ckpt"
        elif token_number == 1:
            ckpt_name = "fine_tuned_David" + str(epoch)+ ".ckpt"
        else:
            ckpt_name = "fine_tuned_cat" + str(epoch)+ ".ckpt"

        save_path = os.path.join(output_dir, ckpt_name)
        save_progress(  text_encoder,
                        [placeholder_token_id],
                        placeholder_token,
                        save_path,
                        safe_serialization= False)
        
        # Do evaluation :
        if (token_number == 0 or token_number == 1):
            model.eval()
            sampler = DPMSolverSampler(model)

            learned_embeds_dict = torch.load(save_path, map_location=device)[placeholder_token]
            tokenizer.add_tokens(placeholder_token)
            token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
            text_encoder.transformer.resize_token_embeddings(len(tokenizer))
            text_encoder.transformer.get_input_embeddings().weight.data[token_id] = learned_embeds_dict
            epoch_dir = os.path.join(output_image_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            for source_num, details in prompt_data.items():
                token_name          = details["token_name"]
                combined_prompts    = details["prompt"] + details["prompt_4_clip_eval"]

                sorce_output_dir = os.path.join(epoch_dir, f"{source_num}")
                os.makedirs(sorce_output_dir, exist_ok = True)
                # Iterate over each prompt for this source
                for prompt_num, prompt_text in enumerate(tqdm(combined_prompts, desc=f"Epoch {epoch} : Processing Prompts")):
                    prompt_output_dir = os.path.join(sorce_output_dir, f"{prompt_num}")
                    os.makedirs(prompt_output_dir, exist_ok=True)

                    # Generate image
                    with autocast("cuda"):
                        base_count = 0
                        for i in range(5):
                            # Set up the conditioning and image generation
                            conditioning    = model.get_learned_conditioning([prompt_text] * 5)
                            # samples_ddim = sampler.sample( S=50,cond = conditioning, batch_size = 5)
                            shape = [4, 512 // 8, 512 // 8]
                            uc = model.get_learned_conditioning(5 * [""])
                            samples_ddim, _ = sampler.sample(  S=50,
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
    parser.add_argument('token_number', type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    args = parser()
    print("token_number :", args.token_number)
    main(args.token_number)
