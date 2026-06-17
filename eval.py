"""Full-frame re-evaluation of trained checkpoints.

train.py's in-run test metric tiles 256px test images (tile 64, pad 8), which
underestimates PSNR by ~1 dB. This script re-evaluates each run's best.pth
full-frame (no tiling), writing <run>/eval/fullframe_metrics.json. Read-only
w.r.t. training; safe to run alongside training.

Usage:
  python eval.py --pattern '*'            # CPU, all runs under results/
  python eval.py --pattern 'gf2_*' --device cuda:0
  python eval.py --pattern 'wv3_*' --device cuda:0 --q-win-size 8   # WV3 = Q8
"""

import argparse
import glob
import json
import os

import torch

from train import build_model, dataset_paths, load_h5
from metrics import calculate_metrics

ROOT = os.path.dirname(os.path.abspath(__file__))
RATIO = 2047.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="*")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="limit test images (smoke test)")
    p.add_argument("--q-win-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(16)

    run_dirs = sorted(glob.glob(os.path.join(ROOT, "results", args.pattern)))
    test_cache = {}
    rows = []
    for run_dir in run_dirs:
        ckpt_path = os.path.join(run_dir, "checkpoints", "best.pth")
        if not os.path.exists(ckpt_path):
            continue
        obj = torch.load(ckpt_path, map_location="cpu")
        run_args = obj["args"]
        dataset = run_args["dataset"]
        kind = run_args["model_kind"]

        if dataset not in test_cache:
            test_cache[dataset] = load_h5(dataset_paths(dataset)["test"], RATIO)
        data = test_cache[dataset]
        n = data["pan"].shape[0] if args.limit is None else min(args.limit, data["pan"].shape[0])

        c_pan = int(data["pan"].shape[1])
        c_ms = int(data["ms"].shape[1])
        model = build_model(kind, c_ms, c_pan, run_args["hidden_dim"], run_args["latent_dim"])
        model.load_state_dict(obj["model"], strict=True)
        model.to(device).eval()

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
            q_win_size=args.q_win_size,
        )
        m = {k: float(v) for k, v in m.items()}
        row = {
            "run": os.path.basename(run_dir),
            "dataset": dataset,
            "model_kind": kind,
            "seed": run_args.get("seed"),
            "ckpt_epoch": obj.get("epoch"),
            "n_test": n,
            "q_win_size": args.q_win_size,
            "fullframe_metrics": m,
        }
        rows.append(row)
        if args.limit is None:
            out_path = os.path.join(run_dir, "eval", "fullframe_metrics.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(row, f, indent=2)
        print(
            f"{row['run']}: ep{row['ckpt_epoch']} "
            f"PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f} "
            f"ERGAS={m['ERGAS']:.4f} Q{args.q_win_size}={m['Q']:.6f}",
            flush=True,
        )

    if not rows:
        print("no checkpoints found")


if __name__ == "__main__":
    main()
