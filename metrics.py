import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")

    # Prefer HWC. If CHW is detected, transpose.
    if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def load_pred_gt(pred_path: str, ref_path: str) -> Tuple[np.ndarray, np.ndarray]:
    pred_mat = sio.loadmat(pred_path)
    ref_mat = sio.loadmat(ref_path)

    if "sr" not in pred_mat:
        raise KeyError(f"Missing 'sr' in prediction file: {pred_path}")

    gt_key = "gt" if "gt" in ref_mat else "I_GT" if "I_GT" in ref_mat else None
    if gt_key is None:
        raise KeyError(f"Missing gt/I_GT in reference file: {ref_path}")

    pred = _to_hwc(pred_mat["sr"]).astype(np.float64)
    gt = _to_hwc(ref_mat[gt_key]).astype(np.float64)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape} for {pred_path}")
    return pred, gt


def rmse_per_band(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    diff = pred - gt
    mse = np.mean(diff * diff, axis=(0, 1))
    return np.sqrt(np.maximum(mse, 0.0))


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float) -> Tuple[float, float]:
    # Returns (mean_band_psnr, global_psnr)
    mse_band = np.mean((pred - gt) ** 2, axis=(0, 1))
    mse_band = np.maximum(mse_band, 1e-12)
    psnr_band = 10.0 * np.log10((data_range ** 2) / mse_band)

    mse_global = float(np.mean((pred - gt) ** 2))
    mse_global = max(mse_global, 1e-12)
    psnr_global = 10.0 * np.log10((data_range ** 2) / mse_global)
    return float(np.mean(psnr_band)), float(psnr_global)


def sam(pred: np.ndarray, gt: np.ndarray) -> float:
    # Spectral Angle Mapper in degrees
    x = pred.reshape(-1, pred.shape[2])
    y = gt.reshape(-1, gt.shape[2])

    dot = np.sum(x * y, axis=1)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    denom = np.maximum(nx * ny, 1e-12)

    cosang = np.clip(dot / denom, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.degrees(np.mean(ang)))


def ergas(pred: np.ndarray, gt: np.ndarray, ratio: float) -> float:
    rmse = rmse_per_band(pred, gt)
    mean_gt = np.mean(gt, axis=(0, 1))
    mean_gt = np.where(np.abs(mean_gt) < 1e-12, 1e-12, mean_gt)
    term = (rmse / mean_gt) ** 2
    return float((100.0 / ratio) * np.sqrt(np.mean(term)))


def q8(pred: np.ndarray, gt: np.ndarray, win_size: int = 8) -> float:
    """Q8: Multi-band universal image quality index (Alparone et al. 2006).
    Processes the full multi-band image jointly rather than per-band then averaging.
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    N = gt.shape[2]
    eps = 1e-12

    # ---------- local means ----------
    means_x = np.empty((N,) + pred.shape[:2], dtype=np.float64)
    means_y = np.empty((N,) + pred.shape[:2], dtype=np.float64)
    for b in range(N):
        means_x[b] = uniform_filter(pred[:, :, b], size=win_size, mode='reflect')
        means_y[b] = uniform_filter(gt[:, :, b], size=win_size, mode='reflect')

    # ---------- local variances ----------
    vx_local = np.empty((N,) + pred.shape[:2], dtype=np.float64)
    vy_local = np.empty((N,) + pred.shape[:2], dtype=np.float64)
    for b in range(N):
        vx_local[b] = uniform_filter(pred[:, :, b] ** 2, size=win_size, mode='reflect') - means_x[b] ** 2
        vy_local[b] = uniform_filter(gt[:, :, b] ** 2, size=win_size, mode='reflect') - means_y[b] ** 2

    # ---------- local cross-covariance (full BxB for each pixel) ----------
    # We compute cov_xy[b1, b2] at every pixel, then sum over channels
    cov_xy_trace = np.zeros(pred.shape[:2], dtype=np.float64)
    for b in range(N):
        # diagonal = local covariance of band b with itself
        cov_local = uniform_filter(pred[:, :, b] * gt[:, :, b], size=win_size, mode='reflect') - means_x[b] * means_y[b]
        cov_xy_trace += cov_local

    # ---------- numerator components ----------
    mu_dot = np.sum(means_x * means_y, axis=0)
    numerator = 4.0 * cov_xy_trace * mu_dot

    # ---------- denominator components ----------
    var_sum = np.sum(vx_local, axis=0) + np.sum(vy_local, axis=0)  # H x W
    mu_norm_sq_x = np.sum(means_x ** 2, axis=0)  # H x W
    mu_norm_sq_y = np.sum(means_y ** 2, axis=0)  # H x W
    den = var_sum * (mu_norm_sq_x + mu_norm_sq_y)  # H x W

    # ---------- final Q8 (spatial mean over valid pixels) ----------
    valid = den > eps
    q_vals = np.zeros_like(den)
    q_vals[valid] = numerator[valid] / den[valid]
    return float(np.mean(q_vals[valid])) if np.any(valid) else 0.0


def calculate_metrics(
    pred,
    gt,
    ratio: float = 4.0,
    data_range: float = 2047.0,
    q_win_size: int = 8,
) -> Dict[str, float]:
    """Compatibility API used by train_rssm.py.

    Accepts torch tensors or numpy arrays with shape [N, C, H, W] or [C, H, W]
    and returns averaged metrics across the batch.
    """
    pred_np = np.asarray(getattr(pred, "detach", lambda: pred)().cpu().numpy() if hasattr(pred, "detach") else pred)
    gt_np = np.asarray(getattr(gt, "detach", lambda: gt)().cpu().numpy() if hasattr(gt, "detach") else gt)

    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred {pred_np.shape}, gt {gt_np.shape}")

    if pred_np.ndim == 3:
        pred_np = pred_np[None, ...]
        gt_np = gt_np[None, ...]
    if pred_np.ndim != 4:
        raise ValueError(f"Expected 4D [N,C,H,W] or 3D [C,H,W], got {pred_np.shape}")

    psnr_list: List[float] = []
    psnr_global_list: List[float] = []
    sam_list: List[float] = []
    ergas_list: List[float] = []
    q_list: List[float] = []

    for i in range(pred_np.shape[0]):
        p = _to_hwc(pred_np[i]).astype(np.float64)
        g = _to_hwc(gt_np[i]).astype(np.float64)
        p_psnr, p_psnr_global = psnr(p, g, data_range=data_range)
        psnr_list.append(p_psnr)
        psnr_global_list.append(p_psnr_global)
        sam_list.append(sam(p, g))
        ergas_list.append(ergas(p, g, ratio=ratio))
        q_list.append(q8(p, g, win_size=q_win_size))

    return {
        "PSNR": float(np.mean(psnr_list)),
        "PSNR_global": float(np.mean(psnr_global_list)),
        "SAM": float(np.mean(sam_list)),
        "ERGAS": float(np.mean(ergas_list)),
        "Q": float(np.mean(q_list)),
    }


def evaluate_pairs(pack_dir: str, ratio: float, data_range: float, q_win_size: int) -> Dict[str, object]:
    index_csv = os.path.join(pack_dir, "index_map_wv3.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Missing index map: {index_csv}")

    rows: List[Dict[str, object]] = []
    with open(index_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            pred_path = os.path.join(pack_dir, row["pred_mat"])
            ref_path = os.path.join(pack_dir, row["ref_mat"])

            pred, gt = load_pred_gt(pred_path, ref_path)
            psnr_mean, psnr_global = psnr(pred, gt, data_range=data_range)
            sample_metrics = {
                "index": idx,
                "PSNR": psnr_mean,
                "PSNR_global": psnr_global,
                "SAM": sam(pred, gt),
                "ERGAS": ergas(pred, gt, ratio=ratio),
                "Q": q8(pred, gt, win_size=q_win_size),
            }
            rows.append(sample_metrics)

    rows = sorted(rows, key=lambda x: x["index"])
    summary = {
        "num_samples": len(rows),
        "PSNR": float(np.mean([r["PSNR"] for r in rows])),
        "PSNR_global": float(np.mean([r["PSNR_global"] for r in rows])),
        "SAM": float(np.mean([r["SAM"] for r in rows])),
        "ERGAS": float(np.mean([r["ERGAS"] for r in rows])),
        "Q": float(np.mean([r["Q"] for r in rows])),
    }
    return {"summary": summary, "per_sample": rows}


def save_outputs(results: Dict[str, object], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "wv3_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, "wv3_metrics_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "PSNR", "PSNR_global", "SAM", "ERGAS", "Q"])
        writer.writeheader()
        for row in results["per_sample"]:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WFANet WV3 results with PSNR/SAM/ERGAS/Q")
    parser.add_argument("--pack-dir", default="eval_wv3_package", help="Path to packaged pred/ref pair directory")
    parser.add_argument("--ratio", type=float, default=4.0, help="Resolution ratio for ERGAS")
    parser.add_argument("--data-range", type=float, default=2047.0, help="Dynamic range for PSNR")
    parser.add_argument("--q-win-size", type=int, default=8, help="Sliding window size for Q index (Q8 -> 8)")
    parser.add_argument("--out-dir", default="eval_wv3_package", help="Output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    pack_dir = args.pack_dir
    if not os.path.isabs(pack_dir) and not os.path.exists(os.path.join(pack_dir, "index_map_wv3.csv")):
        candidate = os.path.join(script_dir, pack_dir)
        if os.path.exists(os.path.join(candidate, "index_map_wv3.csv")):
            pack_dir = candidate

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(script_dir, out_dir)

    results = evaluate_pairs(pack_dir, ratio=args.ratio, data_range=args.data_range, q_win_size=args.q_win_size)
    save_outputs(results, out_dir)

    s = results["summary"]
    print("===== WV3 Evaluation Summary =====")
    print(f"num_samples: {s['num_samples']}")
    print(f"PSNR: {s['PSNR']:.6f}")
    print(f"PSNR_global: {s['PSNR_global']:.6f}")
    print(f"SAM (deg): {s['SAM']:.6f}")
    print(f"ERGAS: {s['ERGAS']:.6f}")
    print(f"Q8: {s['Q']:.6f}")


if __name__ == "__main__":
    main()
