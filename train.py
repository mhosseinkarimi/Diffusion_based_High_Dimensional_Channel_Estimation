import datetime
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.utils import ChannelDataModule
from src.model.diffusion import DiffusionUNet
from src.train.trainer import DDPMLightningModule

warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX`")
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior")


torch.set_float32_matmul_precision('high')  # 'high' or 'medium'; 'ieee' = strict

# New-style per-backend knobs (optional, explicit):
torch.backends.cuda.matmul.fp32_precision = 'high'  # 'high' | 'medium' | 'ieee'
torch.backends.cudnn.conv.fp32_precision  = 'tf32'  # enable TF32 in cuDNN convolutions
def main():
    dm = ChannelDataModule(
        train_path="data/CDL-A/sample_dataset_beamspace.mat",
        file_type="hdf5",
        decompose_mode="real_imag",
        batch_size=256,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        split_val=0.1
    )

    model = DiffusionUNet(
        in_channels=2,
        base_channels=64,
        out_channels=2,
        time_dim=64,
        channel_mults=(1, 2, 2),
        with_cross_attention=False
    )

    diff_module = DDPMLightningModule(
        model=model,
        T=400,
        lr=3e-4,
        wd=1e-2,
        ema_decay=0.9999,
        beta_scheduler='cosine',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer = L.Trainer(
        max_epochs=300,
        precision="16-mixed",
        gradient_clip_val=1.0,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        log_every_n_steps=25,
        val_check_interval=0.25,
        logger=TensorBoardLogger("logs/", name="ddpm_channel_estimation"),
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="ddpm-{epoch:02d}-{val/loss:.4f}",
            )
        ]
    )

    trainer.fit(diff_module, datamodule=dm)
    
    
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()