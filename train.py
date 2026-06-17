"""Equal-budget training for WSRNet and the WFANet baseline.

Trains a model from scratch under a fixed budget (L1 loss, cosine LR, optional
dihedral augmentation) so WSRNet and WFANet can be compared fairly. Select the
model with --model-kind {wsrnet, wfanet}.
"""

import argparse
import json
import math
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from net_torch import HWViT
from model import RSSMHWViTHZ
from metrics import calculate_metrics


ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Phase-0 architecture comparison")
    parser.add_argument("--dataset", choices=["gf2", "qb", "wv3"], required=True)
    parser.add_argument(
        "--model-kind",
        choices=["wsrnet", "wfanet"],
        required=True,
        help="wsrnet = the proposed model; wfanet = the baseline",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--q-win-size", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Dihedral (rot90/flip) geometric augmentation, applied identically to pan/gt/ms/lms; "
                             "uses a dedicated RNG so no-aug runs stay bit-reproducible")
    parser.add_argument("--periodic-save-every", type=int, default=0,
                        help="If >0, also save a checkpoint every N epochs (epoch_NNN.pth) so the true full-frame "
                             "best can be recovered when 64px-val and 256px-fullframe rankings disagree")
    return parser.parse_args()


def dihedral_augment(tensors, rng):
    """Apply one random dihedral-group transform (rot90^k + optional hflip) to a
    list of square NCHW tensors, identically to all so PAN/MS/GT stay aligned.
    rot90/hflip are exact symmetries of nadir satellite imagery and spectrum-
    neutral (no channel mixing), so they cannot bias SAM/spectral metrics."""
    k = rng.randrange(4)
    do_flip = rng.random() < 0.5
    out = []
    for t in tensors:
        if k:
            t = torch.rot90(t, k, dims=(-2, -1))
        if do_flip:
            t = torch.flip(t, dims=(-1,))
        out.append(t.contiguous())
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_h5(path, ratio):
    with h5py.File(path, "r") as f:
        return {
            k: torch.from_numpy(np.asarray(f[k][:], dtype=np.float32) / ratio).float()
            for k in ("pan", "gt", "ms", "lms")
        }


def dataset_paths(dataset):
    base = os.path.join(ROOT, "Dataset", dataset)
    if not os.path.isdir(base) and os.path.isdir(os.path.join(ROOT, "Dataset", dataset.upper())):
        base = os.path.join(ROOT, "Dataset", dataset.upper())  # e.g. Dataset/WV3
    return {
        "train": os.path.join(base, f"train_{dataset}.h5"),
        "val": os.path.join(base, f"valid_{dataset}.h5"),
        "test": os.path.join(base, f"test_{dataset}_multiExm1.h5"),
    }


def build_model(kind, c_ms, c_pan, hidden_dim, latent_dim):
    if kind == "wfanet":
        # Baseline: WFANet (Wavelet-Assisted Multi-Frequency Attention Network).
        return HWViT(
            L_up_channel=c_ms,
            pan_channel=c_pan,
            pan_target_channel=32,
            ms_target_channel=32,
            head_channel=8,
            dropout=0.085,
        )
    if kind == "wsrnet":
        # WSRNet: two-stage, weight-shared scale-recurrent state cell with
        # dw-window-attention ConvGRU gates + level-LL correction; deterministic
        # hidden state (no stochastic z).
        model = RSSMHWViTHZ(
            L_up_channel=c_ms,
            pan_channel=c_pan,
            pan_target_channel=32,
            ms_target_channel=32,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            deterministic_only=True,
            use_conv_gru=True,
            state_conv_type="dw_window_attn",
            state_kernel_size=7,
            freq_state_mode="simple",
            use_wfanet_two_stage=True,
            share_scale_recurrent=True,
            use_level_ll_corr=True,
        )
        # LMS-residual start: zero-init the MS upsampler + fused_weight so the
        # network begins from the upsampled-MS baseline and learns a residual.
        if hasattr(model, "ms_upsample") and len(model.ms_upsample) > 0:
            ms_conv = model.ms_upsample[0]
            if isinstance(ms_conv, nn.Conv2d):
                nn.init.zeros_(ms_conv.weight)
                if ms_conv.bias is not None:
                    nn.init.zeros_(ms_conv.bias)
        if hasattr(model, "fused_weight"):
            model.fused_weight.data.fill_(0.0)
        return model
    raise ValueError(f"Unknown model kind: {kind}")



def forward_model(model, pan, ms, lms):
    out = model(pan=pan, ms=ms, lms=lms)
    if isinstance(out, tuple):
        out = out[0]
    return out


def tiled_forward(model, pan, ms, lms, tile_size=64, pad=8):
    """Run large test images by 64x64 PAN tiles.

    WFANet's full-image attention can OOM on 256x256+ inputs. Tiled inference
    keeps the training protocol unchanged while making final evaluation robust.
    """
    _, _, h_pan, w_pan = pan.shape
    c_out = lms.shape[1]
    scale = h_pan // ms.shape[2]
    tile_size = (tile_size // scale) * scale
    tile_ms = tile_size // scale
    pad = (pad // scale) * scale
    pad_ms = pad // scale

    n_h = (h_pan + tile_size - 1) // tile_size
    n_w = (w_pan + tile_size - 1) // tile_size
    h_pad = n_h * tile_size
    w_pad = n_w * tile_size
    hm_pad = h_pad // scale
    wm_pad = w_pad // scale

    pan_pad = torch.nn.functional.pad(pan, (0, w_pad - w_pan, 0, h_pad - h_pan), mode="reflect")
    lms_pad = torch.nn.functional.pad(lms, (0, w_pad - w_pan, 0, h_pad - h_pan), mode="reflect")
    ms_pad = torch.nn.functional.pad(ms, (0, wm_pad - ms.shape[3], 0, hm_pad - ms.shape[2]), mode="reflect")

    pan_ext = torch.nn.functional.pad(pan_pad, (pad, pad, pad, pad), mode="reflect")
    lms_ext = torch.nn.functional.pad(lms_pad, (pad, pad, pad, pad), mode="reflect")
    ms_ext = torch.nn.functional.pad(ms_pad, (pad_ms, pad_ms, pad_ms, pad_ms), mode="reflect")

    out = torch.zeros(1, c_out, h_pan, w_pan, device=pan.device)
    weight = torch.zeros(1, 1, h_pan, w_pan, device=pan.device)
    for yi in range(n_h):
        for xi in range(n_w):
            py = yi * tile_size + pad
            px = xi * tile_size + pad
            my = yi * tile_ms + pad_ms
            mx = xi * tile_ms + pad_ms
            pan_tile = pan_ext[:, :, py - pad:py + tile_size + pad, px - pad:px + tile_size + pad]
            lms_tile = lms_ext[:, :, py - pad:py + tile_size + pad, px - pad:px + tile_size + pad]
            ms_tile = ms_ext[:, :, my - pad_ms:my + tile_ms + pad_ms, mx - pad_ms:mx + tile_ms + pad_ms]
            pred_tile = forward_model(model, pan_tile, ms_tile, lms_tile).clamp(0.0, 1.0)
            pred_crop = pred_tile[:, :, pad:pad + tile_size, pad:pad + tile_size]
            oy = yi * tile_size
            ox = xi * tile_size
            eh = min(tile_size, h_pan - oy)
            ew = min(tile_size, w_pan - ox)
            out[:, :, oy:oy + eh, ox:ox + ew] += pred_crop[:, :, :eh, :ew]
            weight[:, :, oy:oy + eh, ox:ox + ew] += 1.0
    return out / weight.clamp_min(1.0)


def eval_model(model, data, device, ratio, q_win_size, batch_size):
    model.eval()
    outs = []
    with torch.no_grad():
        n = data["pan"].shape[0]
        if data["pan"].shape[-1] > 128:
            for i in range(n):
                pred = tiled_forward(
                    model,
                    data["pan"][i:i + 1].to(device),
                    data["ms"][i:i + 1].to(device),
                    data["lms"][i:i + 1].to(device),
                )
                outs.append(pred.clamp(0.0, 1.0).cpu())
        else:
            for i in range(0, n, batch_size):
                sl = slice(i, min(i + batch_size, n))
                pred = forward_model(
                    model,
                    data["pan"][sl].to(device),
                    data["ms"][sl].to(device),
                    data["lms"][sl].to(device),
                )
                outs.append(pred.clamp(0.0, 1.0).cpu())
    pred = torch.cat(outs, dim=0)
    metrics = calculate_metrics(
        (pred * ratio).numpy(),
        (data["gt"] * ratio).numpy(),
        ratio=4.0,
        data_range=ratio,
        q_win_size=q_win_size,
    )
    return {k: float(v) for k, v in metrics.items()}


def overall_score(metrics):
    return metrics["PSNR"] + 10.0 * metrics["Q"] - metrics["SAM"] - metrics["ERGAS"]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    ratio = 2047.0
    paths = dataset_paths(args.dataset)
    for split, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split} path: {path}")

    train_data = load_h5(paths["train"], ratio)
    val_data = load_h5(paths["val"], ratio)
    test_data = load_h5(paths["test"], ratio)

    c_pan = int(train_data["pan"].shape[1])
    c_ms = int(train_data["ms"].shape[1])
    run_tag = args.run_tag or (
        f"{args.dataset}_{args.model_kind}_"
        f"s{args.seed}_ep{args.epochs}"
    )
    out_dir = os.path.join(ROOT, "results", run_tag)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    val_dir = os.path.join(out_dir, "val")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_kind, c_ms, c_pan, args.hidden_dim, args.latent_dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"run_tag={run_tag} dataset={args.dataset} model={args.model_kind} "
        f"seed={args.seed} gpu={args.gpu} epochs={args.epochs} batch={args.batch_size}",
        flush=True,
    )
    print(f"train={train_data['pan'].shape} val={val_data['pan'].shape} test={test_data['pan'].shape}", flush=True)
    print(f"model params: {params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    criterion = nn.L1Loss()

    n_train = train_data["pan"].shape[0]
    best_score = -float("inf")
    best_metrics = None
    best_epoch = 0
    history = []
    aug_rng = random.Random(args.seed + 12345) if args.augment else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_steps = 0
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]
            pan = train_data["pan"][idx].to(device)
            gt = train_data["gt"][idx].to(device)
            ms = train_data["ms"][idx].to(device)
            lms = train_data["lms"][idx].to(device)

            if aug_rng is not None:
                pan, gt, ms, lms = dihedral_augment([pan, gt, ms, lms], aug_rng)

            optimizer.zero_grad(set_to_none=True)
            pred = forward_model(model, pan, ms, lms)
            loss = criterion(pred, gt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} step={n_steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1
            if args.max_train_steps is not None and n_steps >= args.max_train_steps:
                break

        progress = epoch / max(1, args.epochs)
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = max(lr, 1e-7)
        for group in optimizer.param_groups:
            group["lr"] = lr

        avg_loss = total_loss / max(1, n_steps)
        entry = {"epoch": epoch, "loss": avg_loss, "lr": lr, "time": time.time() - t0}

        should_val = epoch == 1 or epoch == args.epochs or (args.val_every > 0 and epoch % args.val_every == 0)
        if should_val:
            val_metrics = eval_model(
                model,
                val_data,
                device,
                ratio,
                args.q_win_size,
                args.val_batch_size,
            )
            score = overall_score(val_metrics)
            entry.update({f"val_{k}": v for k, v in val_metrics.items()})
            entry["val_score"] = score
            print(
                f"epoch {epoch:03d}/{args.epochs} loss={avg_loss:.6f} lr={lr:.3e} "
                f"val PSNR={val_metrics['PSNR']:.4f} SAM={val_metrics['SAM']:.4f} "
                f"ERGAS={val_metrics['ERGAS']:.4f} Q4={val_metrics['Q']:.6f} "
                f"score={score:.4f} time={entry['time']:.1f}s",
                flush=True,
            )
            with open(os.path.join(val_dir, f"val_epoch_{epoch:03d}.json"), "w") as f:
                json.dump(entry, f, indent=2)
            if score > best_score:
                best_score = score
                best_metrics = val_metrics
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "metrics": best_metrics,
                        "args": vars(args),
                        "params": params,
                    },
                    os.path.join(ckpt_dir, "best.pth"),
                )
        else:
            print(
                f"epoch {epoch:03d}/{args.epochs} loss={avg_loss:.6f} "
                f"lr={lr:.3e} time={entry['time']:.1f}s",
                flush=True,
            )
        history.append(entry)
        with open(os.path.join(out_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if args.periodic_save_every > 0 and (epoch % args.periodic_save_every == 0 or epoch == args.epochs):
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args), "params": params},
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"),
            )

    print(f"best val epoch={best_epoch} score={best_score:.4f} metrics={best_metrics}", flush=True)
    best = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location="cpu")
    model.load_state_dict(best["model"])
    test_metrics = eval_model(
        model,
        test_data,
        device,
        ratio,
        args.q_win_size,
        args.val_batch_size,
    )
    result = {
        "run_tag": run_tag,
        "dataset": args.dataset,
        "model_kind": args.model_kind,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "params": params,
        "best_val_epoch": best_epoch,
        "best_val_score": best_score,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "q_win_size": args.q_win_size,
    }
    with open(os.path.join(eval_dir, "train_test_metrics_tiled.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("===== TEST =====", flush=True)
    for key in ["PSNR", "SAM", "ERGAS", "Q"]:
        print(f"{key}: {test_metrics[key]:.6f}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
