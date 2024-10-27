import torch

# 載入包含新 token 的模型
new_ckpt_path = '/project/g/r13922043/hw2/checkpoints/model_with_new_token.ckpt'
state_dict = torch.load(new_ckpt_path)

# 嵌入層權重的 key
embedding_key = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
embedding_weights = state_dict["state_dict"].get(embedding_key, None)

# 確認嵌入層權重是否存在
if embedding_weights is None:
    raise KeyError(f"嵌入層權重 '{embedding_key}' 未找到，請檢查路徑。")

# 打印嵌入層權重的形狀
print(f"{embedding_weights.shape}")

# 獲取新 token 的嵌入（新加入的最後一個向量）
new_token_embedding = embedding_weights[-1]
print("新 token 的嵌入向量：")
print(new_token_embedding)