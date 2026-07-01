# SyntheDAS

**Conditional Pixel-Space Diffusion Model for DAS Signal Synthesis & Augmentation**

SyntheDAS trains a 1D diffusion model on Distributed Acoustic Sensing (DAS) fiber-optic data to generate realistic, class-conditional seismic patches. Generated samples are used as synthetic augmentation for a downstream CNN classifier, with a controlled ablation study measuring the impact on classification accuracy.

---

## Results

| Run | Accuracy | F1-macro | F1-car | F1-regular | F1-running | F1-walk |
|-----|----------|----------|--------|------------|------------|---------|
| Baseline (real only) | 0.654 | 0.619 | 0.000 | 0.647 | 0.895 | 0.936 |
| Synthetic BG Mixup | 0.711 | 0.722 | 0.521 | 0.760 | 0.759 | 0.845 |
| Ratio 0.5× synthetic events | 0.858 | 0.860 | 0.752 | 0.921 | 0.862 | 0.905 |
| Ratio 1.0× synthetic events | **0.872** | **0.880** | **0.821** | 0.848 | **0.982** | 0.871 |

Key finding: the baseline fails completely on the "car" class (F1=0.00). Synthetic event injection at 1.0× scale brings car F1 to 0.82 and overall F1-macro from 0.619 → **0.880**.

---

## Architecture

### Diffusion Model — 1D U-Net

```
Input: [B, 8, 16384]   (8 DAS channels × 32.768 s @ 500 Hz)
  │
  ├── Stem Conv1d
  │
  ├── Encoder (4 levels, 96→192→192→384 channels)
  │     Each level: 2× ResBlock + optional self-attention
  │     Dilated convs per level: [1,2] / [1,4] / [1,8] / [1,16]   (WaveNet-style)
  │     Self-attention at deepest level (W=2048) — OOM-safe
  │
  ├── Bottleneck with self-attention
  │
  ├── Decoder (mirror of encoder, skip connections)
  │
  └── Output Conv1d → [B, 8, 16384]
```

**Conditioning (FiLM — Feature-wise Linear Modulation)**

Three signals are projected and summed into every ResBlock:
- `c` — class label (one of: car, regular, running, walk)
- `α` — noise mix level ∈ [0, 1] (0 = clean event, 1 = pure noise)
- `t` — diffusion timestep

During training, class and alpha are independently dropped out with probability 0.1 for classifier-free guidance. At inference, CFG scale = 7.5 sharpens class-specific features.

**Noise schedule & parameterization**
- Schedule: cosine (`squaredcos_cap_v2`), 1 000 timesteps, zero-SNR rescaling
- Prediction target: **v-prediction** (`v = √ᾱ·ε − √(1−ᾱ)·x₀`) — numerically stable at all SNR levels
- Inference: DDIM, 50 steps

**Loss functions**

| Loss | Weight | Purpose |
|------|--------|---------|
| MSE on v-prediction target | 1.0 | Core denoising |
| Multi-scale STFT (1024, 2048) | 0.1 | Spectral fidelity across all frequencies |
| Band-limited STFT 0–128 Hz (1024, 2048) | 1.0 | Emphasises dominant seismic band |
| Derivative L1 | 0.5 | Preserves transient edges / event onsets |
| Min-SNR weighting (γ=5) | — | Prevents high-noise timesteps dominating gradient |

---

### CNN Classifier — DASResNetClassifier

ResNet-34-style 2D CNN trained on DAS patches.

```
Input: [B, 1, 8, 16384]
  │
  ├── Stem: 2× Conv2d(1→64, kernel=(1,7), stride=(1,4))
  │         → compresses T: 16384 → 1024 while preserving channel axis
  │
  ├── Stage 1: 2× BasicBlock(64→64,  stride=(1,2))
  ├── Stage 2: 2× BasicBlock(64→128, stride=(2,2))
  ├── Stage 3: 2× BasicBlock(128→256, stride=(2,2))
  ├── Stage 4: 2× BasicBlock(256→512, stride=(2,2))
  │
  ├── AdaptiveAvgPool2d(1,1)
  ├── Dropout(0.35)
  └── Linear(512 → 4)
```

**Training details**
- Loss: `CrossEntropyLoss` with inverse-frequency class weights
- Optimizer: AdamW (lr=3×10⁻⁴, weight_decay=8.5×10⁻³)
- Scheduler: `CosineAnnealingWarmRestarts` T₀=50
- Augmentation: additive Gaussian noise std=0.3 per batch (normalised space)
- Mixed precision (AMP) + gradient clipping at 1.0

---

## Dataset

**Brno University DAS Dataset** — 4 event classes recorded on a buried fiber-optic cable:

| Class | Description |
|-------|-------------|
| `car` | Vehicle driving overhead — slow horizontal vibration stripes |
| `regular` | Background noise — no structured event |
| `running` | Person running — sharp vertical footstep spikes |
| `walk` | Person walking — slower, heavier spike cadence |

**Patch format:** 8 channels × 16 384 samples (32.768 s @ 500 Hz)

**Recording-level stratified split (70 / 15 / 15)**
- All patches from one recording stay in the same split — no temporal leakage
- Classes with only 2 recordings: rec-1 → train, rec-2 → temporal 50/50 val/test
- Same split shared between diffusion model and CNN — no cross-model leakage
- Decimation: `regular=5%`, `car/running/walk=50%` to balance classes
- Final counts: train 43 296 | val 7 611 | test 5 116 patches

> Update `data.data_dir` in `configs/pixel_diffusion_config.yaml` and `configs/cnn_classifier_config.yaml` to point to your local dataset path.

---

## Installation

```bash
git clone https://github.com/natanelDaniel/das_stable_diffusion.git
cd das_stable_diffusion
pip install -e .
pip install -r requirements.txt
```

**Core dependencies:** `torch>=2.1`, `diffusers>=0.27`, `wandb`, `python-pptx`, `scipy`, `scikit-learn`, `einops`

---

## Project Structure

```
.
├── configs/
│   ├── pixel_diffusion_config.yaml   # Diffusion model — architecture + training + generation
│   └── cnn_classifier_config.yaml    # CNN classifier — training + synthetic data settings
│
├── src/
│   ├── data/
│   │   ├── das_patch_dataset.py          # Raw DAS patch loader
│   │   ├── das_latent_patch_dataset.py   # Patch dataset with mixed-alpha support
│   │   └── splits.py                     # Recording-level stratified split logic
│   ├── models/
│   │   ├── das_diffusion_unet.py         # 1D U-Net with FiLM conditioning
│   │   ├── das_cnn_classifier.py         # DASResNetClassifier (ResNet-34 style)
│   │   └── das_vae_v2.py                 # VAE encoder/decoder (earlier experiment)
│   ├── training/
│   │   ├── diffusion_trainer.py          # Diffusion training loop + EMA + W&B logging
│   │   ├── cnn_trainer.py                # CNN training loop + metrics + W&B logging
│   │   ├── vae_trainer.py                # VAE training loop
│   │   └── losses.py                     # STFT, band-limited STFT, derivative L1 losses
│   └── evaluation/
│       ├── plotting.py                   # Waterfall + spectrogram plot helpers
│       └── diffusion_eval.py             # Class recoverability evaluation
│
├── scripts/
│   ├── train_das_pixel_diffusion.py   # Train the diffusion model
│   ├── train_das_cnn.py               # Train CNN — supports 4 augmentation modes
│   ├── generate_pixel_samples.py      # Generate DAS patches from a checkpoint
│   ├── generate_diffusion_samples.py  # DDIM sampling loop (used by other scripts)
│   ├── eval_cnn_test.py               # Evaluate all CNN runs on test set → ppt_results/
│   ├── plot_real_vs_generated.py      # 4×3 presentation figure: real vs generated
│   ├── plot_alpha_sweep.py            # 5×4 figure: alpha conditioning sweep
│   ├── build_presentation.py          # Build SyntheDAS_presentation.pptx
│   ├── das_viewer.py                  # Interactive Plotly DAS data viewer
│   ├── build_decimation_cache.py      # Pre-cache decimated dataset index
│   └── compute_dataset_stats.py       # Compute normalisation mean/std
│
├── tests/                             # pytest test suite
├── figures/                           # Saved presentation figures
├── ppt_results/                       # CNN evaluation outputs (confusion matrices, ROC, metrics)
└── SyntheDAS_presentation.pptx        # Full 15-slide presentation
```

---

## Workflow

### 1. Train the diffusion model

```bash
python scripts/train_das_pixel_diffusion.py --config configs/pixel_diffusion_config.yaml
```

Checkpoints saved to `checkpoints/das_pixel_diffusion/`. Best checkpoint is `diffusion_best.pt`.  
Training is logged to W&B project `das-stable-diffusion`.

### 2. Generate sample figures

```bash
# 4 classes × [Real | Generated-1 | Generated-2] grid
python scripts/plot_real_vs_generated.py \
    --config configs/pixel_diffusion_config.yaml \
    --ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt

# 5 alpha values × 4 classes grid (event → noise transition)
python scripts/plot_alpha_sweep.py \
    --config configs/pixel_diffusion_config.yaml \
    --ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt
```

### 3. Train the CNN classifier (4 augmentation modes)

Run in order — `synth_ratio_1.0` generates the synthetic cache, `synth_ratio_0.5` reuses half of it:

```bash
# Baseline: real data only
python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run baseline

# BG Mixup: real events + diffusion-generated background noise
python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run synthetic

# Synthetic event injection — generates cache on first run
python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run synth_ratio_1.0

# Reuses the same cache, takes first half per class
python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run synth_ratio_0.5
```

W&B project: `das-cnn-classifier`. Checkpoints saved under `checkpoints/das_cnn_classifier/<run>/`.

### 4. Evaluate all CNN runs

```bash
python scripts/eval_cnn_test.py \
    --config configs/cnn_classifier_config.yaml \
    --runs baseline:checkpoints/das_cnn_classifier/baseline/cnn_best.pt \
           synthetic:checkpoints/das_cnn_classifier/synthetic/cnn_best.pt \
           ratio_0p5:checkpoints/das_cnn_classifier/synth_ratio_0.5/cnn_best.pt \
           ratio_1p0:checkpoints/das_cnn_classifier/synth_ratio_1.0/cnn_best.pt \
    --output ppt_results/
```

Saves confusion matrices, ROC/PR curves, gallery images, `metrics.json`, and `comparison_f1.png` to `ppt_results/`.

### 5. Build the presentation

```bash
python scripts/build_presentation.py
```

Produces `SyntheDAS_presentation.pptx` (15 slides) with all figures and real metric numbers embedded.  
Import into Google Slides: **File → Import Slides → Upload → Import All**.

---

## Augmentation Strategies

### BG Mixup (`--run synthetic`)

At training time, each real event patch is mixed with a diffusion-generated background:
```
patch_out = real_event + α × synthetic_background,   α ~ Uniform(0.0, 0.4)
```
Backgrounds are generated from the `regular` class at 6 alpha levels (200 samples each).

### Synthetic Event Injection (`--run synth_ratio_0.5 / synth_ratio_1.0`)

Full synthetic event patches are generated for all 4 classes across 10 alpha values (500 samples each → 20 000-patch pool). These are concatenated with the real training set:

- `synth_ratio_1.0` — adds synthetic patches equal to the real training set size
- `synth_ratio_0.5` — adds half as many (reuses the first half of each class from the same cache)

Class weights are recomputed over the combined real + synthetic label distribution and fed to a new `WeightedRandomSampler`.

---

## Sampling

Generate patches directly from a trained checkpoint:

```bash
python scripts/generate_pixel_samples.py \
    --config configs/pixel_diffusion_config.yaml \
    --ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt \
    --class_name running \
    --alpha 0.0 \
    --n_samples 4 \
    --cfg_scale 7.5 \
    --steps 50
```

**Alpha semantics:**
- `alpha=0.0` — pure class-conditioned event (strongest class signature)
- `alpha=1.0` — fully noise-dominated (event structure masked)
- Intermediate values smoothly interpolate between the two

---

## Interactive Viewer

```bash
python scripts/das_viewer.py
```

Opens a Plotly Dash app for browsing raw DAS recordings and generated patches.

---

## Tests

```bash
pytest tests/
```

---

## Weights & Biases

| W&B Project | What it tracks |
|-------------|----------------|
| `das-stable-diffusion` | Diffusion model training — loss curves, generated sample grids per epoch |
| `das-cnn-classifier` | CNN runs — accuracy, F1-macro, per-class P/R/F1, confusion matrix, ROC/PR curves |

---

## Citation / Acknowledgements

Dataset: [Brno University DAS Dataset](https://zenodo.org/records/10963552)

Architecture inspired by:
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models*
- Song et al. (2021) — *Denoising Diffusion Implicit Models (DDIM)*
- Salimans & Ho (2022) — *Progressive Distillation / v-prediction*
- Hang et al. (2023) — *Efficient Diffusion Training via Min-SNR Weighting*
- van den Oord et al. (2016) — *WaveNet (dilated causal convolutions)*
