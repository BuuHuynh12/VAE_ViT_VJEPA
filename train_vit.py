"""Train a ViT classifier (supervised, as in the original ViT paper).  S_vit = final [CLS] token.

Objective: cross-entropy on class labels.

    python train_vit.py --epochs 50
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (ViTEncoder, add_common_args, add_vit_args, cosine_lr, get_device, get_loaders,
                    load_resume, save_checkpoint, set_lr, set_seed)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes, img_size=32, patch_size=4, dim=192, depth=6, heads=3):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, 3, dim, depth, heads, use_cls=True)
        self.head = nn.Linear(dim, num_classes)

    def features(self, x):
        return self.encoder(x)[:, 0]  # [CLS] token -> S_vit

    def forward(self, x):
        return self.head(self.features(x))


def main():
    p = add_vit_args(add_common_args(argparse.ArgumentParser()))
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.set_defaults(epochs=50, lr=1e-3, weight_decay=0.05, warmup_epochs=5)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_loader, test_loader, classes = get_loaders(args)
    model = ViTClassifier(len(classes), args.img_size, args.patch_size, args.dim, args.depth, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"ViT params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device: {device}")

    start_epoch = 0
    if args.resume:
        ck = load_resume(args, os.path.join(args.ckpt_dir, "vit.pt"), device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]

    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot_loss = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            set_lr(opt, cosine_lr(step, total, args.warmup_epochs * steps_per_epoch, args.lr))

            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            step += 1

        model.eval()
        te_correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                te_correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"[ViT] epoch {epoch + 1:3d}/{args.epochs} | loss {tot_loss / steps_per_epoch:.4f} | "
              f"train acc {correct / seen:.4f} | test acc {te_correct / len(test_loader.dataset):.4f}")

        save_checkpoint(os.path.join(args.ckpt_dir, "vit.pt"),
                        {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                         "args": vars(args), "classes": classes})
    print(f"saved {args.ckpt_dir}/vit.pt")


if __name__ == "__main__":
    main()
