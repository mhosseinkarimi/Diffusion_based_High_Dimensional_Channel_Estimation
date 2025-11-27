from copy import deepcopy
from typing import Union, List

import lightning as L
import torch
import torch.nn as nn

from src.model.diffusion import DiffusionUNet
from src.train.noise_scheduler import CosineScheduler, LinearScheduler


class DDPMLightningModule(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 T: int = 1000,
                 lr: float = 1e-4,
                 wd : float = 1e-2,
                 ema_decay: float = 0.9999,
                 beta_scheduler: str = 'cosine',
                 beta_start: Union[float, None] = 0.0001,
                 beta_end: Union[float, None] = 0.02,
                 device: str = 'cpu'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        if beta_scheduler == 'cosine':
            self.noise_scheduler = CosineScheduler(num_train_steps=T)
        elif beta_scheduler == 'linear':
            self.noise_scheduler = LinearScheduler(num_train_steps=T, beta_start=beta_start, beta_end=beta_end)
        else:
            raise ValueError(f"Unsupported beta_scheduler: {beta_scheduler}")
        
        # Noise Scgheduler buffers
        betas, alphas, alpha_bars = self.noise_scheduler.betas, self.noise_scheduler.alphas, self.noise_scheduler.alpha_bars
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32), persistent=True)
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32), persistent=True)
        self.register_buffer('alpha_bars', torch.tensor(alpha_bars, dtype=torch.float32), persistent=True)
        
        # EMA 
        self.model_ema = deepcopy(model)
        for param in self.model_ema.parameters():
            param.requires_grad = False
        # Don't assign to self.device (read-only property on nn.Module / LightningModule)
        # Keep a preferred device hint if needed for standalone sampling prior to trainer placement
        self.sample_device_hint = device
        
    @torch.no_grad()
    def ema_update(self, decay=0.999):
        for p_ema, p in zip(self.model_ema.parameters(), self.model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
         
    def forward_sample(self, x0, t, noise):
        """Forward diffusion process: add noise to the clean data x0 at step t."""
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(x0.dtype)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
    
    def training_step(self, batch, batch_idx):
        # Batch is either x0 or (x0, cond)
        cond = None
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x0, cond = batch
        else:
            x0 = batch
        x0 = x0.float()
        
        B = x0.shape[0]
        t = torch.randint(0, self.hparams.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.forward_sample(x0, t, noise)
        
        t_in = t.float()
        
        pred = self.model(x_t, t_in, cond)
        loss = nn.MSELoss()(pred, noise)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # EMA update
        self.ema_update(decay=self.hparams.ema_decay)
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch is either x0 or (x0, cond)
        cond = None
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x0, cond = batch
        else:
            x0 = batch
        x0 = x0.float()
        
        B = x0.shape[0]
        t = torch.randint(0, self.hparams.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.forward_sample(x0, t, noise)
        t_in = t.float()
        pred = self.model_ema(x_t, t_in, cond)
        loss = nn.MSELoss()(pred, noise)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def reverse_sample(self, shape, cond=None, steps=None, use_ema=True):
        """Reverse diffusion process: generate data from noise."""
        eps = 1e-3
        if steps is None:
            steps = self.hparams.T
        model = self.model_ema if use_ema else self.model
        model.eval()
        # Use the actual module device if available (set by Lightning);
        # otherwise fall back to the hint
        dev = getattr(self, 'device', torch.device(self.sample_device_hint))
        x_t = torch.randn(shape, device=dev)
        for t in reversed(range(steps)):
            t_in = torch.full((shape[0],), float(t+1), device=dev)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            pred_noise = model(x_t, t_in, cond)
            mean = (1/torch.sqrt(alpha_t + eps)) * (x_t - (beta_t/torch.sqrt(1 - alpha_bar_t + eps)) * pred_noise)
            
            if t > 0:
                x_t = mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
            else:
                x_t = mean
        return x_t

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                weight_decay=self.hparams.wd)
        return opt        