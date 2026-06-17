# WSRNet — Wavelet Scale-Recurrent Network for Pansharpening

WSRNet treats the wavelet pyramid as a **coarse-to-fine sequence** and fuses the
panchromatic (PAN) and multispectral (MS) inputs across scales with a single
**weight-shared recurrent state cell**, reconstructing each level with the
inverse wavelet transform (IDWT). The recurrent hidden state `h` is
deterministic — there is no stochastic latent variable.

It is designed as a lower-complexity counterpart to
[WFANet](https://github.com/Jie-1203/WFANet) (Wavelet-Assisted Multi-Frequency
**Attention** Network, AAAI 2025): WFANet fuses each frequency subband with an
independent multi-frequency **attention** module, whereas WSRNet replaces those
per-scale attention modules with one **scale-recurrent** state cell shared across
the coarse→fine schedule.

```
PAN, LRMS, LMS
   ├─ feature encode + wavelet decompose (DWT)
   ├─ coarse stage:  shared ConvGRU state cell  ──► IDWT
   └─ fine   stage:  same shared cell (carries h)──► IDWT
   └─ combine + reduce  ─►  HRMS = MS_up + residual
```

Key properties:
- **Weight-shared scale recurrence** — one state cell unrolled coarse→fine (~40% fewer params than an unshared two-stage variant).
- **dw-window-attention ConvGRU gates** + level-wise LL correction.
- **Deterministic hidden state** (no stochastic `z`).
- **Linear-in-pixels**, so full-frame inference on large images is feasible without tiling.

## Install

```bash
pip install -r requirements.txt   # torch>=2.1 (cu121 tested), numpy, h5py, scipy, matplotlib
```

## Data

Reduced-resolution `.h5` datasets, placed as:

```
Dataset/gf2/{train_gf2.h5, valid_gf2.h5, test_gf2_multiExm1.h5}
Dataset/qb/{train_qb.h5, valid_qb.h5, test_qb_multiExm1.h5}
Dataset/WV3/{train_wv3.h5, valid_wv3.h5, test_wv3_multiExm1.h5}
```

Each file holds `pan, ms, lms, gt`. Train/val patches are 64px (PAN) / 16px (MS);
test images are 256px full-frame. Values are normalized by 2047 (11-bit).

## Train

```bash
# WSRNet (main method)
python train.py --dataset gf2 --model-kind wsrnet --augment \
    --epochs 480 --periodic-save-every 60 --gpu 0 --seed 1 --run-tag gf2_wsrnet_s1

# WFANet baseline (same protocol)
python train.py --dataset gf2 --model-kind wfanet --augment \
    --epochs 480 --gpu 1 --seed 1 --run-tag gf2_wfanet_s1

# WV3 is 8-band: add --q-win-size 8
python train.py --dataset wv3 --model-kind wsrnet --augment --q-win-size 8 \
    --epochs 480 --periodic-save-every 60 --gpu 0 --seed 1 --run-tag wv3_wsrnet_s1
```

Equal-budget protocol: from scratch, L1 loss, batch 32, cosine LR (9e-4),
dihedral (rot90/flip) augmentation, deterministic state. Checkpoints go to
`results/<run-tag>/checkpoints/` (`best.pth` by validation score, plus
`epoch_NNN.pth` if `--periodic-save-every > 0`).

## Evaluate (full-frame)

The in-run test metric tiles 256px images and underestimates by ~1 dB; always
re-evaluate full-frame:

```bash
# best.pth, full-frame 256px (GF2/QB use Q4, WV3 use --q-win-size 8)
python eval.py --pattern 'gf2_*' --device cuda:0 --q-win-size 4

# periodic-checkpoint sweep: epoch -> full-frame curve + true best epoch
python eval_sweep.py --pattern 'gf2_*' --device cuda:0 --q-win-size 4
```

`eval_sweep.py` is the tool for cases where 64px-validation and 256px-full-frame
rankings disagree (it reports the full-frame-best epoch vs the val-best epoch and
plots a PSNR-vs-epoch curve).

## Results

Full-frame 256px, 2-seed mean, dihedral augmentation, 480 epochs. GF2/QB use Q4,
WV3 uses Q8. ↑ higher better, ↓ lower better.

### GF2 (4-band)
| Method | PSNR↑ | SAM↓ | ERGAS↓ | Q4↑ |
|---|---|---|---|---|
| WFANet | 50.249 | 0.647 | 0.578 | 0.9091 |
| **WSRNet** | **50.464** | **0.637** | **0.565** | **0.9126** |

### WV3 (8-band)
| Method | PSNR↑ | SAM↓ | ERGAS↓ | Q8↑ |
|---|---|---|---|---|
| WFANet | 38.999 | 2.934 | 2.167 | 0.8761 |
| **WSRNet** | **39.163** | **2.872** | **2.120** | **0.8765** |

### QB (4-band)
| Method | PSNR↑ | SAM↓ | ERGAS↓ | Q4↑ |
|---|---|---|---|---|
| **WFANet** | **38.765** | **4.358** | **3.522** | **0.8448** |
| WSRNet | 38.369 | 4.466 | 3.687 | 0.8387 |

WSRNet overtakes WFANet on **GF2 (+0.215 dB)** and **WV3 (+0.164 dB)** across all
metrics. On **QB** WSRNet is behind (−0.40 dB); QB has the smallest training set
and the finest, most PAN-dominated high-frequency content, where the recurrent
state overfits the 64px training patches (its full-frame quality peaks early,
around epoch 180–300, then regresses while validation keeps rising — use
`eval_sweep.py` to locate that point).

## Acknowledgements

The wavelet transforms and the WFANet baseline (`net_torch.HWViT`) follow
[WFANet](https://github.com/Jie-1203/WFANet). WSRNet keeps that wavelet dataflow
but replaces the per-subband attention fusion with a shared scale-recurrent
state cell.
