# Reproducing "Vision Transformers Need Registers" at Small Scale

Token norms, register ablation and attention maps on Imagenette.

**Author:** Abhinav Saxena (IDD CSE & AI, Roll No. 24CS2003)
**Paper reproduced:** T. Darcet, M. Oquab, J. Mairal, P. Bojanowski, *Vision Transformers Need Registers*, ICLR 2024 ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588))

---

## What this project does

Large Vision Transformers can develop a small set of **high-norm "artifact" tokens** in low-information (often background) patches, which makes attention maps spiky. The paper's fix is to append a few learnable **register tokens** that the network can use as scratch memory and that are discarded at the output.

This repo reproduces the core mechanism on one Kaggle Tesla T4 GPU:

1. **Part 1: artifacts in a pre-trained model.** Token-norm analysis of the pre-trained DINOv2 ViT-g/14.
2. **Part 2: registers in a small ViT.** A compact ViT written from scratch in PyTorch with a configurable number of registers, trained on Imagenette with **0 vs 4 registers over 5 seeds**.
3. **Part 3: analysis.** Supervised accuracy, linear probe, token-norm statistics, and last-layer `[CLS]` attention maps.

The register mechanism (model, training loop, norm and attention analysis) is implemented by the author in PyTorch. Standard building blocks (`nn.Linear`, `nn.Conv2d`, `AdamW`, `torch.hub`) are used as allowed.

---

## Key results

### Part 1: pre-trained DINOv2 ViT-g/14 (no registers, 64 validation images, 16,384 patch tokens)

| Statistic | Value |
|---|---|
| Median patch-token norm | 49.09 |
| Max patch-token norm | 542.80 (about 11x the median) |
| Tokens with norm > 150 | **2.88%** (about 470 of 16,384) |
| Tokens > 2.5x their image median | 2.88% (same fraction, both rules agree) |

The distribution is clearly bimodal (Fig. 1), and the outliers are isolated tokens mostly on grass, sky and plain backgrounds (Fig. 2), in line with the paper (which reports roughly 2%). In the register variant (`dinov2_vitg14_reg`) the norm maps look smooth with no flagged tokens. This register-model result is a **visual check only**; a numeric outlier fraction was not computed.

### Part 2: small ViT on Imagenette (mean ± std over 5 seeds, 40 epochs)

| Model | Supervised val acc (%) | Linear probe val acc (%) | Seeds |
|---|---|---|---|
| Small ViT, 0 registers | 76.50 ± 0.57 | 76.08 ± 0.41 | 5 |
| Small ViT, 4 registers | 77.11 ± 0.52 | 76.61 ± 0.71 | 5 |
| **Difference (4 − 0)** | **+0.60** | **+0.54** | |

- Registers add only **768 parameters** (4 × D, D = 192) and 4 extra tokens per layer (257 → 261 tokens, about 1.6%).
- Registers were ahead in **4 of 5 seeds** for both metrics.
- The gain is about the size of the seed standard deviation: Welch p = 0.12 (supervised) and 0.19 (probe); paired p = 0.26 and 0.32. We therefore report it as a **consistent small positive trend that is not statistically significant** with 5 seeds.

<details>
<summary>Per-run results</summary>

| Seed | Registers | Supervised val acc (%) | Linear probe (%) | Train time (s) |
|---|---|---|---|---|
| 42 | 0 | 77.30 | 76.76 | 344.79 |
| 42 | 4 | 76.54 | 75.62 | 353.94 |
| 43 | 0 | 76.48 | 76.05 | 347.51 |
| 43 | 4 | 76.74 | 76.28 | 353.76 |
| 44 | 0 | 76.69 | 75.77 | 347.51 |
| 44 | 4 | 77.45 | 76.99 | 353.87 |
| 45 | 0 | 75.72 | 76.05 | 350.16 |
| 45 | 4 | 77.81 | 77.50 | 362.37 |
| 46 | 0 | 76.33 | 75.75 | 355.78 |
| 46 | 4 | 76.99 | 76.66 | 362.27 |

</details>

### Artifact and attention statistics (seed 42, 1000 validation images)

| Model | Outlier fraction (> 2.5x image median) | Max / median norm | CLS attention entropy (normalised) | CLS attention on registers |
|---|---|---|---|---|
| 0 registers | 0.0000 | 1.759 | 0.744 | n/a |
| 4 registers | 0.0001 | 1.702 | 0.777 | 0.0065 |

Neither small model developed high-norm artifacts (max/median stays below 1.8, versus about 11 for ViT-g/14). This is consistent with the paper's claim that artifacts appear only in large, long-trained models. Our experiment did not vary model size or training length, so it is a consistency check rather than a test of that claim. These statistics come from a single seed and are indicative only.

## Method in brief

For `N` patches embedded to `D` dimensions, the standard ViT input is `Z0 = [x_cls; x_1; ...; x_N] + E_pos`. With `R` registers `r_1..r_R` (learnable, shared across images, **no positional embedding**):

```
Z0_reg = [Z0 ; r_1 ; ... ; r_R]        # (1 + N + R) x D
```

All tokens pass through `L` pre-norm transformer blocks. After the last block the registers are dropped and the classifier reads only `[CLS]`:

```
y_hat = W * LN(Z_L[CLS]) + b
```

Registers read from and write to image tokens in every layer, but no loss term supervises them directly.

**Artifact measurement.** A final-layer patch token `z_i` is an outlier if `||z_i||_2 > 150` (the absolute threshold used in the paper for ViT-g) or `||z_i||_2 > 2.5 x median_j ||z_j||_2` within the same image.

---

## Setup used

| Item | Value |
|---|---|
| Hardware / software | Kaggle notebook, 1x NVIDIA Tesla T4, PyTorch 2.11 (CUDA 12.8) |
| Data | Imagenette `imagenette2-160`: 10 classes, 9,469 train / 3,925 val images |
| Input pipeline | Decoded once, resized and center-cropped to 144 px, cached as `uint8` on GPU; training uses random 128 px crop + horizontal flip; validation uses center 128 px crop |
| Model | Custom ViT: patch 8, width 192, depth 6, 3 heads, 256 patch tokens, 2.758 M params (2.759 M with 4 registers) |
| Optimiser | AdamW, lr 1e-3, weight decay 0.05 (not applied to biases, norms, `[CLS]`, positional and register params) |
| Schedule | 3 warm-up epochs then cosine decay, 40 epochs total |
| Regularisation | Label smoothing 0.1, gradient clipping 1.0, batch size 128, fp16 autocast |
| Register init | Truncated normal, sigma = 0.02, no positional embedding |
| Attention | Fused scaled-dot-product attention for training; explicit float32 softmax in the last block to extract attention maps |
| Linear probe | Frozen `[CLS]` features (after final LayerNorm), standardised, single linear layer, 300 full-batch AdamW steps |
| Part 1 model | `dinov2_vitg14` via `torch.hub`, 64 random val images resized to 224 px; norms taken from the final block output before the last LayerNorm |
| Seeds | 42, 43, 44, 45, 46 |
| Run time | About 6 minutes per run (345 to 362 s, roughly 9 s per epoch) |

---

## How to run

> **TODO before publishing:** replace `paper_reproduction.ipynb` and the folder names below with the real ones in your repository.

### On Kaggle

1. Create a new Kaggle notebook, enable **GPU (T4)** and **Internet**.
2. Upload or import `paper_reproduction.ipynb`.
3. Run all cells. If Imagenette is not attached under `/kaggle/input`, the notebook downloads `imagenette2-160` automatically.
4. The notebook runs, in order:
   - data caching and sanity checks (token count 257 vs 261, parameter count),
   - Part 1: DINOv2 ViT-g/14 norm analysis and figures,
   - Part 2: training 0 and 4 registers for seeds 42 to 46,
   - Part 3: linear probe, statistics (Welch and paired tests), attention and norm figures.

### Locally / Colab

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install torch torchvision numpy scipy pandas matplotlib
jupyter notebook YOUR-NOTEBOOK.ipynb
```

A CUDA GPU is strongly recommended. The DINOv2 ViT-g/14 weights are about 4.2 GB and are downloaded by `torch.hub` on first use.

---


---

## Reproducibility notes

- Seeds are fixed (42 to 46), but GPU kernels were **not forced to be deterministic** and fp16 autocast is used, so repeated runs can differ slightly across hardware.
- Part 1 uses 64 images and one measurement point.
- Attention maps and the artifact statistics table use the seed-42 models only.

## Limitations

- **Scale.** The small ViT (2.76 M parameters, 40 epochs, under 10k images) is far smaller than the models in the paper and did not develop artifacts, so the artifact-removal benefit of registers could not be tested there. Showing it directly needs a larger model, longer training, or self-supervised training.
- **Accuracy gain is within noise.** +0.60 / +0.54 points with Welch p > 0.1.
- **No sweep over register count** (only R = 0 vs 4), and no positional-embedding variant.
- **No dense-task evaluation** (segmentation, depth, object discovery), which is where the paper reports its larger gains.
- The linear probe uses features of a *supervised* backbone, whereas the paper probes self-supervised features.
- Imagenette has 10 classes, so accuracies saturate and differences are small.
- The register-variant DINOv2 result in Part 1 is a visual check only.

## Possible extensions

- Sweep R in {0, 1, 2, 4, 8, 16} and test positional embeddings on registers.
- Train a larger ViT longer, ideally self-supervised, and check whether artifacts appear and whether registers remove them.
- Add dense evaluation and a localisation metric for the attention maps.

---

## References and code consulted

1. T. Darcet, M. Oquab, J. Mairal, P. Bojanowski, "Vision Transformers Need Registers," ICLR 2024. arXiv:2309.16588.
2. A. Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021. arXiv:2010.11929.
3. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2024. arXiv:2304.07193.
4. M. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," ICCV 2021. arXiv:2104.14294.
5. G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024. arXiv:2309.17453.
6. M. Sun et al., "Massive Activations in Large Language Models," arXiv:2402.17762, 2024.
7. J. Howard, "Imagenette," fast.ai. https://github.com/fastai/imagenette

Code consulted / used:

- [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2): used **only** to load pre-trained DINOv2 and DINOv2-with-registers weights through `torch.hub` for Part 1.
- [`fastai/imagenette`](https://github.com/fastai/imagenette): dataset source.

<!-- TODO: add any other repository or blog you consulted -->


## License

This project is licensed under the [MIT License](LICENSE).
