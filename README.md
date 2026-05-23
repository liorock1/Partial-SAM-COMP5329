# Partial-SAM-COMP5329: Efficiency Frontiers in Flat-Minima Optimization

> **COMP5329 / COMP4329 Deep Learning — Assignment 2**
> University of Sydney · DLCC 2026

A systematic empirical investigation into reducing the computational cost of Sharpness-Aware Minimization (SAM) without sacrificing generalization. We explore two orthogonal axes of reduction **temporal scheduling** and **structural sparsity** and combine them in a **joint ablation** that achieves ~50% FLOPs savings with minimal accuracy loss on CIFAR-100.

---

## Overview

SAM improves generalization by seeking flatter minima in the loss landscape, but requires two forward-backward passes per update — doubling the training cost of SGD. This project asks:

> *Can we apply SAM to only part of the training (temporal) or part of the network (structural), and still retain most of its generalization benefit?*

We evaluate this on **ResNet-18 / CIFAR-100** across three ablation tracks:

| Track | Question | Contributor |
|---|---|---|
| Temporal | Apply SAM only in the last α fraction of epochs | Vaibhavi Shivanna |
| Structural | Apply SAM perturbation only to selected residual blocks | Xiaohe Bu |
| **Joint** | **Combine temporal scheduling + structural sparsity** | **Lior Sabbagh** |

---

## Key Results

| Configuration | Test Accuracy | Sharpness (ρ) | FLOPs Saved |
|---|---|---|---|
| SGD baseline | 75.71% | 0.0929 | 50.0% |
| Full SAM (α=1.0) | 64.10% | 0.2033 | 0.0% |
| Temporal α=0.25 | 71.85% | 0.2083 | 37.0% |
| Structural First-1 (B1) | 71.65% | 0.1845 | 49.3% |
| Structural Last-2 (B3+B4) | 71.59% | 0.1887 | 3.2% |
| **Joint α=0.25, first1** | **71.52%** | **0.2332** | **49.8%** |
| **Joint α=0.25, first2** | **71.05%** | **0.2327** | **49.2%** |

The joint configurations sit on the **Pareto frontier** of the efficiency–generalization trade-off, matching the accuracy of the best individual strategies while maximizing compute savings.

---

## Repository Structure

```
partial-sam/
│
├── PartialSam_Joint.ipynb          # Lior — Joint ablation (temporal + structural)
├── PartialSam_Temporal.ipynb       # Vaibhavi — Temporal ablation
├── PartialSam_Structural.ipynb     # Xiaohe — Structural ablation
│
├── results/
│   ├── results_temporal.csv
│   ├── results_structural.csv
│   ├── results_joint.csv
│   └── figure1_pareto.png
│
└── README.md
```

---

## Notebooks

### `PartialSam_Joint.ipynb` — Joint Ablation (this repo's primary notebook)

Implements and evaluates the **joint Partial-SAM** strategy, which combines:
- **Temporal scheduling** (α=0.25): SGD runs for the first 75% of training; PartialSAM activates for the final 25%
- **Structural sparsity**: the SAM perturbation ascent step is restricted to early residual blocks only (B1 or B1+B2), covering as little as **1.3% of model parameters**

The descent step always updates **all parameters**, preserving the full model's optimization trajectory while the sharpness penalty is applied cheaply.

Two configurations are evaluated:
- `first1` (B1 only, 1.3% of params) — 71.52% accuracy, 49.8% FLOPs saved
- `first2` (B1+B2, 6.0% of params) — 71.05% accuracy, 49.2% FLOPs saved

The notebook also generates all tables and the Pareto frontier figure (Figure 1) from the combined results of all three ablation tracks.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Model | ResNet-18 (CIFAR-adapted: 3×3 stem, no maxpool) |
| Dataset | CIFAR-100 (50k train / 10k test) |
| Epochs | 50 |
| Batch size | 128 |
| Learning rate | 0.1 with cosine annealing |
| Momentum | 0.9 |
| Weight decay | 5×10⁻⁴ |
| SAM ρ | 0.05 |
| Seed | 42 |
| Hardware | NVIDIA Tesla T4 (Google Colab) |

---

## Method Summary

**Temporal Partial-SAM**
```
optimizer(t) = SAM    if t ≥ T·(1 − α)
               SGD    otherwise
```

**Structural Partial-SAM**
The perturbation ε is computed only over a selected layer mask M ⊆ {B1, B2, B3, B4}:
```
ε_l = ρ · ∇_l L(w) / ‖∇_M L(w)‖    if l ∈ M
      0                               otherwise
```
The SGD descent step still updates all parameters.

**Joint Partial-SAM** combines both: SGD for epochs 1–37, then PartialSAM on selected early layers for epochs 38–50.

---

## How to Run

All notebooks are designed for **Google Colab with a T4 GPU**.

1. Open the desired notebook in Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run Sections 1–7 (shared setup) — everyone runs these
4. Run your assigned section (8 = Temporal, 9 = Structural, 10 = Joint)
5. Download results via Section 11

To regenerate all figures and tables, run Section 12 in `PartialSam_Joint.ipynb` after uploading all three results CSVs.

---

## References

1. Foret et al. (2021). *Sharpness-Aware Minimization for Efficiently Improving Generalization.* ICLR 2021.
2. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
3. Krizhevsky (2009). *Learning Multiple Layers of Features from Tiny Images.*
4. Liu et al. (2022). *LookSAM: Alleviating the Expensive Computation in SAM.* CVPR 2022.
5. Zhu et al. (2025). *Implicit Bias of SAM: On the Escaping Dynamics from Sharp Minima.* ICLR 2025 (Spotlight).
6. Du et al. (2022). *Sharpness-Aware Training for Free.* NeurIPS 2022. (SSAM)
7. Du et al. (2022). *Efficient Sharpness-Aware Minimization for Improved Training of Neural Networks.* ICLR 2022. (ESAM)
8. Mueller et al. (2023). *Normalization Layers Are All That Sharpness-Aware Minimization Needs.* NeurIPS 2023.

---

## Authors

- **Vaibhavi Shivanna** — Temporal ablation
- **Xiaohe Bu** — Structural ablation  
- **Lior Sabbagh** — Joint ablation & visualization

*COMP5329 Deep Learning, University of Sydney, 2026*
