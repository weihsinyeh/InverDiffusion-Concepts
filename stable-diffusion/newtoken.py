import torch
import numpy as np

ckpt_path = "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
state_dict = torch.load(ckpt_path, map_location="cpu")

# locate the parameter
embedding_key = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
embedding_weights = state_dict["state_dict"].get(embedding_key, None)

if embedding_weights is None:
    raise KeyError(f"未找到嵌入層權重，請檢查嵌入層名稱。")

vocab_size, embedding_dim = embedding_weights.shape
print(f"embedding dimension : {embedding_weights.shape}")

new_token_embedding = torch.randn(embedding_dim) * embedding_weights.std()
print(f"new token embedding shape{new_token_embedding.shape}")

new_embeddings = torch.cat([embedding_weights, new_token_embedding.unsqueeze(0)], dim=0)
state_dict["state_dict"][embedding_key] = new_embeddings

new_ckpt_path = "/project/g/r13922043/hw2/checkpoints/model_with_new_token.ckpt"
torch.save(state_dict, new_ckpt_path)
print(f"new mode is saved to: {new_ckpt_path}")