"""Periodic-checkpoint full-frame sweep.

For each matched run, full-frame-evaluates every periodic checkpoint
(epoch_NNN.pth) plus best.pth, so we can see the epoch -> full-frame-metric
curve and locate the TRUE full-frame-best epoch. This is the tool for the QB
case where 64px-val keeps rising while 256px-full-frame regresses: val-best
(best.pth) and full-frame-best can be different epochs.

Read-only w.r.t. training. Writes results to <run>/eval/periodic_sweep.json
and, if matplotlib is available, a PSNR-vs-epoch PNG per dataset.

Usage:
  python eval_sweep.py --pattern 'qb_*' --device cuda:0
  python eval_sweep.py --pattern 'qb_*' --device cuda:0 --q-win-size 4
  # WV3 must pass --q-win-size 8
"""

import argparse
import glob
import json
import os
import re

import torch

from train import build_model, dataset_paths, load_h5
from metrics import calculate_metrics

ROOT = os.path.dirname(os.path.abspath(__file__))
RATIO = 2047.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="*", help="run-dir glob under results/")
    p.add_argument("--device", default="cpu")
    p.add_argument("--q-win-size", type=int, default=4, help="Q window (GF2/QB=4, WV3=8)")
    p.add_argument("--limit", type=int, default=None, help="limit test images (smoke)")
    p.add_argument("--no-plot", action="store_true", help="skip the matplotlib PNG even if available")
    return p.parse_args()


def fullframe_eval(model, data, n, device, q_win_size):
    preds = []
    with torch.no_grad():
        for i in range(n):
            out = model(
                pan=data["pan"][i:i + 1].to(device),
                ms=data["ms"][i:i + 1].to(device),
                lms=data["lms"][i:i + 1].to(device),
            )
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.clamp(0.0, 1.0).cpu())
    pred = torch.cat(preds, dim=0)
    m = calculate_metrics(
        (pred * RATIO).numpy(),
        (data["gt"][:n] * RATIO).numpy(),
        ratio=4.0,
        data_range=RATIO,
        q_win_size=q_win_size,
    )
    return {k: float(v) for k, v in m.items()}


def list_checkpoints(run_dir):
    """Return [(epoch:int, source:str, path:str)] sorted by epoch; best.pth last."""
    ckpts = []
    for p in glob.glob(os.path.join(run_dir, "checkpoints", "epoch_*.pth")):
        m = re.search(r"epoch_(\d+)\.pth$", p)
        if m:
            ckpts.append((int(m.group(1)), "periodic", p))
    ckpts.sort(key=lambda x: x[0])
    best = os.path.join(run_dir, "checkpoints", "best.pth")
    if os.path.exists(best):
        ckpts.append((-1, "best", best))  # epoch filled in after load
    return ckpts


def maybe_plot(per_dataset, device_str, no_plot):
    if no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable — skipping PNG, JSON + table still written)")
        return
    for dataset, runs in per_dataset.items():
        plt.figure(figsize=(7, 5))
        for run, sweep in runs.items():
            pts = [(s["epoch"], s["PSNR"]) for s in sweep if s["source"] == "periodic"]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            label = run.replace(f"{dataset}_", "").replace("_ep480", "")
            plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("epoch")
        plt.ylabel("full-frame PSNR (dB)")
        plt.title(f"{dataset}: full-frame PSNR vs epoch (periodic ckpts)")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        out = os.path.join(ROOT, "results", f"periodic_sweep_{dataset}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  plot -> {out}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(16)

    run_dirs = sorted(glob.glob(os.path.join(ROOT, "results", args.pattern)))
    test_cache = {}
    per_dataset = {}

    for run_dir in run_dirs:
        ckpts = list_checkpoints(run_dir)
        if not ckpts:
            continue
        run = os.path.basename(run_dir)
        print(f"\n=== {run} ({len([c for c in ckpts if c[1]=='periodic'])} periodic + best) ===", flush=True)

        sweep = []
        for epoch, source, path in ckpts:
            obj = torch.load(path, map_location="cpu")
            run_args = obj["args"]
            dataset = run_args["dataset"]
            kind = run_args["model_kind"]
            ep = obj.get("epoch", epoch)

            if dataset not in test_cache:
                test_cache[dataset] = load_h5(dataset_paths(dataset)["test"], RATIO)
            data = test_cache[dataset]
            n = data["pan"].shape[0] if args.limit is None else min(args.limit, data["pan"].shape[0])

            model = build_model(kind, int(data["ms"].shape[1]), int(data["pan"].shape[1]),
                                run_args["hidden_dim"], run_args["latent_dim"])
            model.load_state_dict(obj["model"], strict=True)
            model.to(device).eval()
            m = fullframe_eval(model, data, n, device, args.q_win_size)
            sweep.append({"epoch": ep, "source": source, **m})
            print(f"  {source:8s} ep{ep:>3} PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f} "
                  f"ERGAS={m['ERGAS']:.4f} Q{args.q_win_size}={m['Q']:.6f}", flush=True)
            del model

        # locate true full-frame best (periodic ckpts only) vs val-best (best.pth)
        periodic = [s for s in sweep if s["source"] == "periodic"]
        ff_best = max(periodic, key=lambda s: s["PSNR"]) if periodic else None
        val_best = next((s for s in sweep if s["source"] == "best"), None)
        result = {
            "run": run, "dataset": dataset, "model_kind": kind,
            "seed": run_args.get("seed"), "q_win_size": args.q_win_size,
            "n_test": n, "sweep": sweep,
            "fullframe_best": ff_best, "val_best": val_best,
        }
        os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
        with open(os.path.join(run_dir, "eval", "periodic_sweep.json"), "w") as f:
            json.dump(result, f, indent=2)
        if ff_best and val_best:
            print(f"  -> fullframe-best ep{ff_best['epoch']} PSNR={ff_best['PSNR']:.4f} | "
                  f"val-best ep{val_best['epoch']} PSNR={val_best['PSNR']:.4f} | "
                  f"gap={ff_best['PSNR']-val_best['PSNR']:+.4f}", flush=True)
        per_dataset.setdefault(dataset, {})[run] = sweep

    if not per_dataset:
        print("no checkpoints found for pattern")
        return
    maybe_plot(per_dataset, args.device, args.no_plot)


if __name__ == "__main__":
    main()
