import torch
from src.model.diffusion import DiffusionUNet
from src.train.trainer import DDPMLightningModule

backbone = DiffusionUNet(
    in_channels=2,
    base_channels=64,
    out_channels=2,
    time_dim=256
)

ckpt_path = "logs/ddpm_channel_estimation/version_5/checkpoints/ddpm-epoch=254-val/loss=0.1022.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
module = DDPMLightningModule(model=backbone)
module.load_state_dict(ckpt["state_dict"], strict=True)  # should succeed
module.eval()