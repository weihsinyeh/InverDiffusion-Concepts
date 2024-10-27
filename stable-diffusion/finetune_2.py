import torch
import os
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.util import instantiate_from_config
from transformers import CLIPTokenizer
import torch.optim as optim
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import itertools
from omegaconf import OmegaConf
import numpy as np
import random
def load_model_from_config2(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    
    # Check for the embeddings key
    embedding_key = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"

    # Check if the embeddings key exists in the loaded state dict
    if embedding_key in sd:
        loaded_embedding_size = sd[embedding_key].size(0)
        current_embedding_size = model.cond_stage_model.transformer.get_input_embeddings().weight.size(0)

        if loaded_embedding_size != current_embedding_size:
            print(f"Size mismatch detected: loaded {loaded_embedding_size}, current {current_embedding_size}. Resizing embeddings.")
            model.cond_stage_model.transformer.resize_token_embeddings(loaded_embedding_size)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model, pl_sd['tokenizer']

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

# Create dataset and dataloader
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions, tokenizer, transform=None):
        self.image_dir = image_dir
        self.captions = captions
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, f"0{idx}.jpg")).convert("RGB")
        image = self.transform(image)
        caption = self.captions[idx]
        #tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        return {"pixel_values": image, "input_ids": caption}

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
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)
        '''
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        '''
        example["input_ids"] = text
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

config_path = "./configs/stable-diffusion/v1-inference.yaml"
checkpoint_path = "./models/ldm/stable-diffusion-v1/model.ckpt"
output_dir = "/project/g/r13922043/hw2/checkpoints"
image_folder = "/project/g/r13922043/hw2_data/textual_inversion/0"
placeholder_token = "<new1>"

captions = [    f"This is {placeholder_token} in a specific style.",
                f"An example of {placeholder_token}.",
                f"Representation of {placeholder_token}.",
                f"A scene with {placeholder_token}.",
                f"Artwork of {placeholder_token}.",]

# Load Model Configuration and Checkpoint
config = OmegaConf.load(config_path)
model = load_model_from_config(config, checkpoint_path)

# Initialize CLIPTokenizer and add placeholder token
tokenizer = model.cond_stage_model.tokenizer
num_added_tokens = tokenizer.add_tokens([placeholder_token])
if num_added_tokens == 0:
    raise ValueError(f"The tokenizer already contains the token {placeholder_token}. Please use a unique token.")

# Load Dataset
train_data = TextualInversionDataset(data_root=image_folder,  tokenizer=tokenizer)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

# Convert token to IDs
token_ids = tokenizer.encode(placeholder_token)
placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

# Load text encoder
text_encoder = model.cond_stage_model

# Resize embeddings to include the new token
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Embedding layer size: {text_encoder.transformer.get_input_embeddings().weight.shape[0]}")
text_encoder.transformer.resize_token_embeddings(len(tokenizer))
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Embedding layer size: {text_encoder.transformer.get_input_embeddings().weight.shape[0]}")

optimizer = optim.AdamW(    text_encoder.transformer.get_input_embeddings().parameters(),   lr=5e-3)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Training loop
for epoch in range(20):  # Adjust epochs as needed
    model.train()
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        batch["pixel_values"] = batch["pixel_values"].to(device)
        print(batch["input_ids"])
        
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
        print(batch["input_ids"])
        print("token_id : ",token_id)
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
            print("Gradients exist.")
            
            # Create a mask to identify non-placeholder tokens
            index_grads_to_zero = torch.arange(len(text_encoder.tokenizer)) != placeholder_token_id
            
            # Set gradients for non-placeholder tokens to zero
            model_embeddings.data[index_grads_to_zero, :] = 0  # Clear gradients for non-placeholder tokens
        else:
            print("No gradients were computed.")

        # Perform optimizer step
        optimizer.step()
        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    model.cond_stage_model.tokenizer = tokenizer
    ckpt_name = "fine_tuned_" + str(epoch)+ ".ckpt"
    new_ckpt_path = os.path.join(output_dir, ckpt_name)
    checkpoint = {"state_dict": model.state_dict()}
    checkpoint["tokenizer"] = tokenizer
    # Save the model's state_dict inside the checkpoint dictionary
    torch.save(checkpoint, new_ckpt_path)

'''

state_dict = torch.load(checkpoint_path)

# Locate the parameter for the original embedding weights
embedding_key = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
embedding_weights = state_dict["state_dict"].get(embedding_key, None)

if embedding_weights is None:
    raise KeyError(f"cannot fine embedding weights.")

vocab_size, embedding_dim = embedding_weights.shape
print(f"Original embedding dimension: {embedding_weights.shape}")

updated_embeddings = text_encoder.transformer.get_input_embeddings().weight.data.clone()
state_dict["state_dict"][embedding_key] = updated_embeddings
tokenizer_key = "cond_stage_model.tokenizer"
state_dict[tokenizer_key] = tokenizer
# Assume your trained text_encoder is already on the CPU and in 'text_encoder'
# Save the state_dict of your new text_encoder
text_encoder_state_dict = text_encoder.state_dict()
# Replace the relevant part in the original state dict
state_dict["state_dict"]["cond_stage_model.transformer"] = text_encoder_state_dict

new_ckpt_path = os.path.join(output_dir, "fine_tuned.ckpt")
torch.save(state_dict, new_ckpt_path)
print(f"New model is saved to: {new_ckpt_path}")
'''
config = OmegaConf.load(config_path)
model, tokenizer = load_model_from_config2(config, new_ckpt_path)
tokens = tokenizer('<new1>', return_tensors="pt", padding=True, truncation=True)
print("tokens",tokens)
tokenizer = model.cond_stage_model.tokenizer
tokens = tokenizer('<new1>', return_tensors="pt", padding=True, truncation=True)
print("tokens",tokens)