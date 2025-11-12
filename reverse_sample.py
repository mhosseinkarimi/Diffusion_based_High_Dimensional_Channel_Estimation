import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from src.train.trainer import DDPMLightningModule
from src.model.diffusion import DiffusionUNet
from src.model.diffusion import DiffusionUNet

ckpt_path = "logs/ddpm_channel_estimation/version_5/checkpoints/ddpm-epoch=254-val/loss=0.1022.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
backbone = DiffusionUNet(
    in_channels=2,
    base_channels=64,
    out_channels=2,
    time_dim=256
)
module = DDPMLightningModule(model=backbone)
module.load_state_dict(ckpt["state_dict"], strict=True)  # should succeed
module.eval()

x_reversed = module.reverse_sample(
    shape=(1, 2, 64, 64),
    steps=1000,
    use_ema=True
).cpu().numpy()

print("Reversed sample shape:", x_reversed.shape)
print("Reversed sample stats: min {:.4f}, max {:.4f}, mean {:.4f}, std {:.4f}".format(
    x_reversed.min(), x_reversed.max(), x_reversed.mean(), x_reversed.std()
))
plt.imshow(20*np.log10(abs(x_reversed[0, 0])), cmap='viridis')
plt.colorbar()
plt.show()