import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import MedicalImageFolder
from diffusion_model import SimpleUnet, linear_beta_schedule


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Pretrain unconditional diffusion prior on medical images (16-bit)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="prior.pth")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=0, help="If >0, overrides epochs with total training steps")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action='store_true')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Data
    dataset = MedicalImageFolder(args.data_dir, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Model
    model = SimpleUnet(dropout_rate=args.dropout_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Schedule buffers
    betas = linear_beta_schedule(args.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == 'cuda'))

    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return sqrt_alphas_cumprod[t].reshape(-1,1,1,1) * x_start + sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1) * noise, noise

    model.train()
    step = 0
    with tqdm(total=(args.steps if args.steps>0 else args.epochs*len(loader)), desc="Pretraining") as pbar:
        for epoch in range(max(1, args.epochs)):
            for batch in loader:
                if args.steps>0 and step>=args.steps:
                    break
                x0 = batch.to(device)  # (B,1,H,W) in [-1,1]
                t = torch.randint(0, args.timesteps, (x0.size(0),), device=device).long()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_amp):
                    x_t, noise = q_sample(x0, t)
                    pred = model(x_t, t, condition=x0)  # unconditional: we can use x0 as a weak condition to stabilize; or use zeros_like(x0)
                    loss = F.mse_loss(pred, noise)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                step += 1
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            if args.steps>0 and step>=args.steps:
                break

    torch.save(model.state_dict(), args.output)
    print(f"Saved prior weights to {args.output}")


if __name__ == "__main__":
    set_seed(42)
    main()