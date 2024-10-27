import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np

image_dir = "/project/g/r13922043/hw2_data/textual_inversion/0"

def load_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
    
    return torch.stack(images)

class TextualInversionModel(nn.Module):
    def __init__(self, base_model, token_size):
        super(TextualInversionModel, self).__init__()
        self.base_model = base_model
        self.token_embedding = nn.Parameter(torch.randn(token_size, 768))  # 假設嵌入大小是768
        
    def forward(self, input_ids):
        # 使用嵌入進行模型的前向傳播
        embedded_input = self.token_embedding[input_ids]
        return self.base_model(embedded_input)

def train_model(model, images, epochs=100, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 損失函數
    
    for epoch in range(epochs):
        for img in images:
            optimizer.zero_grad()
            output = model(img)  # 假設 img 已經轉換成合適的輸入格式
            loss = criterion(output, img)  # 計算損失
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

base_model  = "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
state_dict = torch.load(ckpt_path, map_location="cpu")
autoencoder.load_state_dict(state_dict["state_dict"], strict=False)

# 生成潜在编码 latent_codes
latent_codes = []
for image_tensor in images:
    image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        latent_code = autoencoder.encode(image_tensor).sample()
        latent_codes.append(latent_code)

print(f"成功生成的潜在编码数量: {len(latent_codes)}")

# 初始化“dog”作为新概念词
opt = lambda: None  # 使用匿名对象存储配置
opt.init_word = "dog"
config.model.params.personalization_config.params.initializer_words[0] = opt.init_word

# 获取嵌入层的权重
embedding_key = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
embedding_weights = state_dict["state_dict"][embedding_key]

# 创建新词嵌入，并将其添加到模型的嵌入层
new_token_embedding = torch.randn(embedding_weights.shape[1]) * embedding_weights.std()
new_embeddings = torch.cat([embedding_weights, new_token_embedding.unsqueeze(0)], dim=0)
state_dict["state_dict"][embedding_key] = new_embeddings

# 保存更新后的模型检查点
new_ckpt_path = "/project/g/r13922043/hw2/checkpoints/model_with_new_token.ckpt"
torch.save(state_dict, new_ckpt_path)
print(f"New model with 'dog' concept saved to: {new_ckpt_path}")

# 设置优化器和损失函数
new_token_embedding = new_token_embedding.clone().requires_grad_(True)
optimizer = optim.Adam([new_token_embedding], lr=0.005)
loss_fn = torch.nn.MSELoss()

# 训练参数
num_epochs = 5000
time_steps = np.linspace(0, 1, num_epochs)

