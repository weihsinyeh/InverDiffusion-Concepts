from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from digit_dataloader import digit_dataset
from DDPM import DDPM 
from model import Unet
# Usage:
# $ python PB1_training.py

# https://github.com/TeaPearce/Conditional_Diffusion_MNIST
def rm_tree(pth: Path):
    if pth.is_dir():
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()


mean, std = [0.4632, 0.4669, 0.4195], [0.1979, 0.1845, 0.2082]
mnistm_train_set = digit_dataset(
    root='../hw2_data/digits/mnistm/data/',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),  # 添加标准化
    ]),
    label_csv=['../hw2_data/digits/mnistm/train.csv'],
    dataset_id = 0
)

svhn_train_set = digit_dataset(
    root='../hw2_data/digits/svhn/data/',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),  
    ]),
    label_csv=['../hw2_data/digits/svhn/train.csv'],
    dataset_id = 1
)
train_set = torch.utils.data.ConcatDataset([mnistm_train_set, svhn_train_set])

batch_size = 128
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=6)

num_epochs = 200
n_T = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
n_features = 128
ckpt_path = Path('../P1_ckpt')
tb_path = Path('../P1_tb')

rm_tree(ckpt_path)
rm_tree(tb_path)

ckpt_path.mkdir(exist_ok=True)
tb_path.mkdir(exist_ok=True)

ddpm = DDPM(
    model=Unet(
        in_channels=3,
        n_features=n_features,
        n_classes=20
    ),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1
).to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()
writer = SummaryWriter(tb_path)

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    ddpm.train()

    optim.param_groups[0]['lr'] = lr * (1 - epoch / num_epochs)

    for x, digit_label, dataset_label in tqdm(train_loader):
        with torch.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.float16):
            x = x.to(device, non_blocking=True)
            digit_label = digit_label.to(device, non_blocking=True)
            dataset_label = dataset_label.to(device, non_blocking=True)
            # Adjust digit_label based on dataset_label
            digit_label = digit_label + (10 * dataset_label)  # If dataset_label is 1, this adds 10 to digit_label

            loss = ddpm(x, digit_label)

        scaler.scale(loss).backward()
        scaler.step(optim)  # replaces optim.step()
        scaler.update()
        optim.zero_grad()

    ddpm.eval()
    with torch.no_grad():
        n_samples = 30
        for gw in [0, 0.5, 2]:
            x_gen, x_gen_store = ddpm.sample(
                n_samples, (3, 28, 28), device, guide_w=gw)
            grid = make_grid(x_gen * -1 + 1, nrow=3)
            writer.add_image(f'DDPM results/w={gw:.1f}', grid, epoch)
            grid = make_grid(x_gen, nrow=3)
            writer.add_image(f'DDPM results wo inv/w={gw:.1f}', grid, epoch)

    torch.save(ddpm.state_dict(), ckpt_path / f"{epoch}_ddpm.pth")
