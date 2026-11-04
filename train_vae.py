"""Train a convolutional VAE.  S_vae = mu(x), the posterior mean.

Objective: reconstruct pixels + keep q(z|x) close to N(0, I)
    loss = ||x - x_hat||^2 + beta * KL(q(z|x) || N(0, I))

    python train_vae.py --epochs 30
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from common import (add_common_args, cosine_lr, denormalize, get_device, get_loaders, load_resume,
                    save_checkpoint, set_lr, set_seed)


class ConvVAE(nn.Module):
    def __init__(self, img_size=32, latent_dim=128, ch=(64, 128, 256)):
        super().__init__()
        assert img_size % 8 == 0
        self.s = img_size // 8
        c1, c2, c3 = ch

        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())

        self.encoder = nn.Sequential(down(3, c1), down(c1, c2), down(c2, c3), nn.Flatten())
        feat = c3 * self.s * self.s
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, feat)
        self.decoder = nn.Sequential(
            up(c3, c2), up(c2, c1),
            nn.ConvTranspose2d(c1, 3, 4, 2, 1), nn.Tanh(),  # images are in [-1, 1]
        )
        self.c3 = c3

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.c3, self.s, self.s)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()  # reparameterization trick
        return self.decode(z), mu, logvar


def vae_loss(x, recon, mu, logvar):
    B = x.size(0)
    rec = F.mse_loss(recon, x, reduction="sum") / B
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return rec, kl


def main():
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl_warmup_epochs", type=int, default=5, help="linearly ramp beta from 0 (avoids posterior collapse)")
    p.set_defaults(epochs=30, lr=1e-3, weight_decay=0.0, warmup_epochs=1)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader, classes = get_loaders(args)
    model = ConvVAE(args.img_size, args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"VAE params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device: {device}")

    start_epoch = 0
    if args.resume:
        ck = load_resume(args, os.path.join(args.ckpt_dir, "vae.pt"), device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]

    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot_rec = tot_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            set_lr(opt, cosine_lr(step, total, args.warmup_epochs * steps_per_epoch, args.lr))
            beta = args.beta * min(1.0, step / max(1, args.kl_warmup_epochs * steps_per_epoch))

            recon, mu, logvar = model(x)
            rec, kl = vae_loss(x, recon, mu, logvar)
            loss = rec + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_rec += rec.item()
            tot_kl += kl.item()
            step += 1

        # ---- eval
        model.eval()
        te_rec = te_kl = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                rec, kl = vae_loss(x, recon, mu, logvar)
                te_rec += rec.item() * len(x)
                te_kl += kl.item() * len(x)
        n_test = len(test_loader.dataset)
        print(f"[VAE] epoch {epoch + 1:3d}/{args.epochs} | beta {beta:.2f} | "
              f"train rec {tot_rec / steps_per_epoch:8.2f} kl {tot_kl / steps_per_epoch:7.2f} | "
              f"test rec {te_rec / n_test:8.2f} kl {te_kl / n_test:7.2f}")

        save_checkpoint(os.path.join(args.ckpt_dir, "vae.pt"),
                        {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                         "args": vars(args), "classes": classes})

    # ---- reconstructions (top: input, bottom: recon) and random samples
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x[:16].to(device)
        recon, _, _ = model(x)
        save_image(denormalize(torch.cat([x, recon])), os.path.join(args.out_dir, "vae_recon.png"), nrow=16)
        z = torch.randn(64, args.latent_dim, device=device)
        save_image(denormalize(model.decode(z)), os.path.join(args.out_dir, "vae_samples.png"), nrow=8)
    print(f"saved {args.ckpt_dir}/vae.pt, {args.out_dir}/vae_recon.png, {args.out_dir}/vae_samples.png")


if __name__ == "__main__":
    main()
