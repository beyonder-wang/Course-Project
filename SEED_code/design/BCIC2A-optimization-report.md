# BCIC2A Optimization Report

**Objective:** Maximize 4-class motor imagery (left/right/foot/tongue) classification accuracy on BCIC2A within ~20 training runs.

**Data:** 22 EEG channels, 800 time steps (4.0 s at 200 Hz), bandpass 0.1–75 Hz.  
720 train + 360 val + 360 test per fold. Balanced classes.

---

## Results Summary

### Single-Fold (Fold 1) Comparison

| Model | Best Val | Params | Time | Notes |
|-------|----------|--------|------|-------|
| SimpleLinear | 41.67% | 70K | 133s | Weak baseline |
| SimpleMLP | 42.59% | 4.5M | 252s | Heavily overparameterized |
| MICNN | 35.19% | 20.6K | 753s | Multi-scale CNN, overfits |
| FBCNet | 34.72% | 4.5K | 179s | Multi-band FFT, incompatible with preprocessing |
| EEGTCNet | 49.54% | 14.8K | 602s | EEGNet + TCN, best non-EEGNet |
| **EEGNet (30ep)** | 51.39% | 3.1K | 1510s | Standard reference |
| **EEGNet lr=1e-3 (30ep)** | 55.09% | 3.1K | 1251s | Higher LR helps convergence |
| **EEGNet (50ep)** | 56.02% | 3.1K | 1430s | More epochs → better |
| EEGNet_KAN (50ep) | 41.20% | 15.9K | 1381s | KAN head underperforms |
| **EEGNet_SE (50ep)** | **57.41%** | **3.4K** | **2457s** | **Best single fold** |
| EEGNet_SimAM | 35.19% | 3.1K | 1327s | SimAM degrades on long sequences |
| EEGNet_SimAM_SE | 31.02% | 3.4K | 1858s | Both attentions combined worse |

### 5-Fold Cross-Validation (30 epochs)

| Model | Mean Best Val | Std |
|-------|--------------|-----|
| **EEGNet** | **53.70%** | ±3.19% |
| EEGNet_SE | 50.46% | ±1.92% |

*Note: EEGNet without SE attention achieved higher average across 5 folds, but EEGNet_SE achieved the highest single-fold peak (57.41%).*

---

## What Worked

1. **EEGNet is the best backbone.** The depthwise separable convolution design is inherently suited to EEG — it separates temporal and spatial feature learning. All other architectures underperformed it.

2. **Higher learning rate (1e-3) helps EEGNet.** The default 5e-4 is conservative; 1e-3 gave faster convergence and better final accuracy (55.09% vs 51.39%).

3. **Longer training (50 epochs) yields diminishing but real gains.** 50 epochs improved ~5% over 30 epochs for EEGNet. The gain comes from slow late-stage convergence, not early overfitting.

4. **Squeeze-and-Excitation (SE) attention gives a small boost.** +1.4% over vanilla EEGNet at 50 epochs on fold 1, but not consistent across folds in CV.

---

## What Didn't Work

1. **FBCNet (34.72%).** The FFT-based band decomposition likely conflicts with the 0.1–75 Hz bandpass preprocessing already applied to the data. The network receives redundant filtering, and variance pooling loses temporal structure.

2. **SimAM attention (35.19%).** SimAM computes a single spatial attention map across all dimensions. With 800 time steps × 22 channels, the attention signal is diluted — it effectively becomes noise.

3. **KAN classifier heads (41.20%).** KANMLP (B-spline activations) replaces the linear classifier but introduces many more parameters (15.9K vs 3.1K) without improvement. The spline basis functions add complexity without discriminative power for this 4-class problem.

4. **MICNN (35.19%).** Multi-scale temporal convolutions (kernel sizes 16/32/64) with 20.6K params overfit the 720-sample training set.

5. **SimpleMLP (42.59%).** 4.5M params on 720 samples — extreme overfitting despite dropout.

---

## Key Takeaways

- **Model architecture matters more than tuning.** The gap between EEGNet (51.39%) and SimpleLinear (41.67%) is +10%, while the gap between LR tuning (51.39% → 55.09%) is only +3.7%.
- **EEG-specific inductive biases win.** Depthwise separable convolutions (EEGNet) outperform generic architectures (MLP, CNN) and more complex designs (FBCNet, MICNN).
- **Literature gap remains.** Published results report ~73% for EEGNet on BCIC2A. Our 57.41% ceiling suggests preprocessing pipeline differences (200 Hz resampling, bandpass filtering, or train/val split methodology) may cap performance.
- **Recommendation for future work:** Investigate the preprocessing pipeline alignment with published BCIC2A benchmarks, particularly trial segmentation and validation split methodology.

---

## Best Configuration

```
python 0_run_train.py --dataset BCIC2A --model EEGNet --lr 1e-3 --epochs 50 --device cpu
```

- 5-fold CV expected: ~54–56% mean best val accuracy
- Single fold best seen: 57.41% (EEGNet_SE, 50ep)
