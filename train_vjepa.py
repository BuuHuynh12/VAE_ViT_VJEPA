"""Train V-JEPA on images (an image = a video with a single frame, so tubelets become 2D patches).
S_vjepa = mean-pooled output of the (EMA) target encoder.

Objective: predict the *representations* (not pixels) of masked patches.
    context encoder  E_theta      sees only the visible patches
    target encoder   E_bar        = EMA(E_theta), sees all patches, no gradient
    predictor        P_phi        context tokens + [MASK] tokens (with positions) -> predicted target tokens
    loss = || P_phi(E_theta(x_visible), pos_masked) - sg(LN(E_bar(x)))[masked] ||_1

Labels are NOT used for training; they are only used for the kNN monitor.

    python train_vjepa.py --epochs 100
"""
import argparse
import copy
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (Block, ViTEncoder, add_common_args, add_vit_args, cosine_lr, extract_features,
                    gather_tokens, get_datasets, get_device, get_loaders, init_weights, knn_accuracy,
                    load_resume, save_checkpoint, set_lr, set_seed, sincos_pos_embed_2d)


class Predictor(nn.Module):
    """Narrow transformer that predicts target-encoder tokens at masked positions."""

    def __init__(self, grid, enc_dim, pred_dim=96, depth=4, heads=3):
        super().__init__()
        self.embed = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.register_buffer("pos_embed", sincos_pos_embed_2d(pred_dim, grid)[None], persistent=False)
        self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, enc_dim)
        self.apply(init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx, keep_idx, mask_idx):
        B = ctx.size(0)
        pos = self.pos_embed.expand(B, -1, -1)
        x = self.embed(ctx) + gather_tokens(pos, keep_idx)
        m = self.mask_token.expand(B, mask_idx.size(1), -1) + gather_tokens(pos, mask_idx)
        x = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x[:, -mask_idx.size(1):])


class MultiBlockMask:
    """V-JEPA multi-block masking on a 2D patch grid.

    Each batch uses one of two strategies (as in V-JEPA):
      short-range: union of 8 small blocks (scale 0.15 each)
      long-range : union of 2 large blocks (scale 0.70 each)  -> ~90% of patches masked
    The number of kept / masked tokens is truncated to the batch minimum so tensors stay rectangular.
    """

    def __init__(self, grid, configs=((8, 0.15), (2, 0.7)), aspect=(0.75, 1.5), min_keep=4):
        self.grid = grid
        self.configs = configs
        self.log_aspect = (math.log(aspect[0]), math.log(aspect[1]))
        self.min_keep = min_keep

    def _block(self, scale):
        area = scale * self.grid ** 2
        ar = math.exp(random.uniform(*self.log_aspect))
        h = max(1, min(self.grid, round(math.sqrt(area * ar))))
        w = max(1, min(self.grid, round(math.sqrt(area / ar))))
        top = random.randint(0, self.grid - h)
        left = random.randint(0, self.grid - w)
        return top, left, h, w

    def __call__(self, batch_size, device):
        num_blocks, scale = random.choice(self.configs)
        masks = []
        for _ in range(batch_size):
            for _ in range(50):
                m = torch.zeros(self.grid, self.grid, dtype=torch.bool)
                for _ in range(num_blocks):
                    t, l, h, w = self._block(scale)
                    m[t:t + h, l:l + w] = True
                if (~m).sum() >= self.min_keep:
                    break
            masks.append(m.flatten())
        masks = torch.stack(masks)  # (B, N) True = masked

        n_keep = int((~masks).sum(1).min())
        n_mask = int(masks.sum(1).min())
        # random subset of each sample's visible / masked positions (wanted ones get scores in [0,1))
        r = torch.rand(masks.shape)
        keep_idx = (r + masks.float() * 2).argsort(1)[:, :n_keep]
        mask_idx = (r + (~masks).float() * 2).argsort(1)[:, :n_mask]
        return keep_idx.to(device), mask_idx.to(device)


def main():
    p = add_vit_args(add_common_args(argparse.ArgumentParser()))
    p.add_argument("--pred_dim", type=int, default=96)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=5, help="kNN monitor every N epochs (0 = off)")
    p.set_defaults(epochs=100, lr=5e-4, weight_decay=0.05, warmup_epochs=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_loader, test_loader, classes = get_loaders(args)

    encoder = ViTEncoder(args.img_size, args.patch_size, 3, args.dim, args.depth, args.heads).to(device)
    target_encoder = copy.deepcopy(encoder).requires_grad_(False)
    predictor = Predictor(encoder.grid, args.dim, args.pred_dim, args.pred_depth, args.heads).to(device)
    masker = MultiBlockMask(encoder.grid)

    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"V-JEPA encoder {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M | "
          f"predictor {sum(p.numel() for p in predictor.parameters()) / 1e6:.2f}M | device: {device}")

    # loader without augmentation for the kNN monitor
    if args.eval_every:
        train_eval, _, _ = get_datasets(args.data, args.data_dir, args.img_size, train_aug=False)
        knn_train_loader = torch.utils.data.DataLoader(train_eval, args.batch_size, shuffle=True,
                                                       num_workers=args.num_workers)

    start_epoch = 0
    if args.resume:
        ck = load_resume(args, os.path.join(args.ckpt_dir, "vjepa.pt"), device)
        encoder.load_state_dict(ck["encoder"])
        target_encoder.load_state_dict(ck["target_encoder"])
        predictor.load_state_dict(ck["predictor"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]

    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        predictor.train()
        tot_loss = tot_std = 0.0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            set_lr(opt, cosine_lr(step, total, args.warmup_epochs * steps_per_epoch, args.lr))
            keep_idx, mask_idx = masker(x.size(0), device)

            with torch.no_grad():
                h = target_encoder(x)
                h = F.layer_norm(h, (h.size(-1),))
                tgt = gather_tokens(h, mask_idx)

            z = encoder(x, keep_idx)
            pred = predictor(z, keep_idx, mask_idx)
            loss = F.l1_loss(pred, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # EMA update of the target encoder, momentum ramps ema_start -> ema_end
            m = args.ema_start + (args.ema_end - args.ema_start) * step / total
            with torch.no_grad():
                for pt, pc in zip(target_encoder.parameters(), encoder.parameters()):
                    pt.mul_(m).add_(pc.detach(), alpha=1 - m)

            tot_loss += loss.item()
            tot_std += h.mean(1).std(0).mean().item()  # ~0 would mean representation collapse
            step += 1

        msg = (f"[V-JEPA] epoch {epoch + 1:3d}/{args.epochs} | loss {tot_loss / steps_per_epoch:.4f} | "
               f"feat std {tot_std / steps_per_epoch:.4f} | ema {m:.4f}")
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            target_encoder.eval()
            fn = lambda im: target_encoder(im).mean(1)
            tr_f, tr_y = extract_features(fn, knn_train_loader, device, max_samples=10000)
            te_f, te_y = extract_features(fn, test_loader, device)
            msg += f" | kNN acc {knn_accuracy(tr_f, tr_y, te_f, te_y, num_classes=len(classes)):.4f}"
        print(msg)

        save_checkpoint(os.path.join(args.ckpt_dir, "vjepa.pt"),
                        {"target_encoder": target_encoder.state_dict(), "encoder": encoder.state_dict(),
                         "predictor": predictor.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                         "args": vars(args), "classes": classes})
    print(f"saved {args.ckpt_dir}/vjepa.pt")


if __name__ == "__main__":
    main()
