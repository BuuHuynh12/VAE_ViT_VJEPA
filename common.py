"""Shared pieces for the three models: data, ViT building blocks, schedules, kNN eval.

All three scripts load data through `get_datasets` with the same preprocessing,
so S_vae, S_vit and S_vjepa are computed from exactly the same images.
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)  # images live in [-1, 1] for every model


# ----------------------------------------------------------------------------- utils
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ----------------------------------------------------------------------------- args
def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--data", default="cifar10", choices=["cifar10", "folder"],
                   help="cifar10, or an ImageFolder with <data_dir>/train and <data_dir>/val")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_dir", default="./checkpoints")
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="0 = full train set; >0 = subset (quick smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="continue from <ckpt_dir>/<model>.pt (keep the same --epochs / --lr as the original run)")
    return p


def add_vit_args(p: argparse.ArgumentParser):
    # Same defaults in train_vit.py and train_vjepa.py -> identical encoder architecture
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=3)
    return p


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, state):
    """Write to a temp file then rename, so an interrupt never leaves a half-written checkpoint."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(args, path, device):
    """Load a checkpoint written by save_checkpoint (must contain 'opt' and 'epoch')."""
    if not os.path.exists(path):
        raise SystemExit(f"--resume: {path} not found")
    ck = torch.load(path, map_location=device)
    if "opt" not in ck or "epoch" not in ck:
        raise SystemExit(f"--resume: {path} has no optimizer state (saved by an older version); "
                         "it can be used for visualize.py but not resumed")
    old = ck["args"]
    for k in ("epochs", "lr", "batch_size", "warmup_epochs"):
        if old.get(k) != getattr(args, k):
            print(f"warning: --{k} is {getattr(args, k)} but the checkpoint was trained with {old.get(k)}; "
                  "the lr schedule will not match the original run")
    print(f"resuming from {path} at epoch {ck['epoch']}/{args.epochs}")
    return ck


# ----------------------------------------------------------------------------- data
def get_transform(img_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_datasets(name, data_dir, img_size, train_aug=True):
    """Returns (train_set, test_set, class_names). Same images for all 3 models."""
    t_train = get_transform(img_size, train=train_aug)
    t_test = get_transform(img_size, train=False)
    if name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=t_train)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=t_test)
    elif name == "folder":
        train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=t_train)
        test = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=t_test)
    else:
        raise ValueError(name)
    return train, test, train.classes


def get_loaders(args):
    train, test, classes = get_datasets(args.data, args.data_dir, args.img_size)
    if args.max_train_samples > 0:
        idx = torch.randperm(len(train))[: args.max_train_samples].tolist()
        train = torch.utils.data.Subset(train, idx)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)
    return train_loader, test_loader, classes


# ----------------------------------------------------------------------------- ViT blocks
def sincos_pos_embed_2d(dim, grid):
    """Fixed 2D sin-cos positional embedding (as in MAE / I-JEPA / V-JEPA). -> (grid*grid, dim)"""
    assert dim % 4 == 0
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing="ij")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))

    def emb(pos):
        out = pos.flatten()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    return torch.cat([emb(ys), emb(xs)], dim=1)


def gather_tokens(x, idx):
    """x: (B, N, D), idx: (B, K) -> (B, K, D)"""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ViTEncoder(nn.Module):
    """Plain ViT encoder. Used by the supervised ViT (use_cls=True) and by V-JEPA (use_cls=False)."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=192, depth=6, heads=3, use_cls=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.grid = self.patch_embed.grid
        self.num_patches = self.patch_embed.num_patches
        self.dim = dim
        self.register_buffer("pos_embed", sincos_pos_embed_2d(dim, self.grid)[None], persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls else None
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(init_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, keep_idx=None):
        """keep_idx: (B, K) indices of visible patches (V-JEPA context). None = all patches."""
        x = self.patch_embed(x) + self.pos_embed
        if keep_idx is not None:
            x = gather_tokens(x, keep_idx)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def extract_features(fn, loader, device, max_samples=0):
    feats, labels = [], []
    n = 0
    for x, y in loader:
        feats.append(fn(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
        n += len(y)
        if max_samples and n >= max_samples:
            break
    feats, labels = torch.cat(feats), torch.cat(labels)
    if max_samples:
        feats, labels = feats[:max_samples], labels[:max_samples]
    return feats, labels


@torch.no_grad()
def knn_accuracy(train_f, train_y, test_f, test_y, k=20, temperature=0.07, num_classes=None):
    """Weighted cosine kNN (DINO-style). A cheap probe of how semantic a feature space is."""
    num_classes = num_classes or int(train_y.max()) + 1
    train_f = F.normalize(train_f, dim=1)
    test_f = F.normalize(test_f, dim=1)
    correct = 0
    for i in range(0, len(test_f), 1000):
        sim = test_f[i:i + 1000] @ train_f.T
        w, idx = sim.topk(k, dim=1)
        votes = torch.zeros(len(idx), num_classes)
        votes.scatter_add_(1, train_y[idx], (w / temperature).exp())
        correct += (votes.argmax(1) == test_y[i:i + 1000]).sum().item()
    return correct / len(test_f)
