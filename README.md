# Self-Pruning Neural Network

A feed-forward neural network that **learns to prune its own weights during training** using learnable gate parameters and L1 sparsity regularisation — no post-training pruning step required.

Built for the Tredence Analytics AI Engineer case study.

---

## Project Structure

```
.
├── self_pruning_nn.py   # Main implementation (all parts in one file)
├── report.md            # Explanation + results report (pre-filled structure)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── outputs/             # Generated after running (plots + auto-report)
    ├── gate_distributions.png
    ├── training_curves.png
    └── report.md        # Auto-generated with real numbers
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full training experiment (3 lambda values, 30 epochs each)
python self_pruning_nn.py
```

Training logs print to stdout. All plots and the final report are written to `./outputs/`.

Expected runtime: ~10–20 min on CPU, ~3–5 min on GPU.

---

## Key Components

### `PrunableLinear` (Part 1)

A custom linear layer that replaces `nn.Linear`. Each weight element has a
corresponding learnable `gate_score`. During the forward pass:

```python
gates         = sigmoid(gate_scores)        # squash to (0, 1)
pruned_weights = weight * gates             # element-wise mask
output        = F.linear(x, pruned_weights, bias)
```

Gradients flow through both `weight` and `gate_scores` automatically via autograd.

### Sparsity Loss (Part 2)

```
L_total = CrossEntropy(logits, targets) + λ × Σ sigmoid(gate_scores)
```

The L1 norm of the gates (their sum, since gates ≥ 0) applies constant-magnitude
gradient pressure toward zero — unlike L2 which allows small non-zero gates to
survive.

### Training & Evaluation (Part 3)

Three runs with λ ∈ {1e-5, 1e-4, 5e-4}. After training:
- Sparsity level = % of gates below threshold (1e-2)
- Gate distribution plot shows bimodal pattern (spike at 0 + retained cluster)

---

## Results (Example — replace with actual after running)

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:----------:|:-------------:|:------------:|
| 1e-5       | ~52%          | ~5%          |
| 1e-4       | ~49%          | ~45%         |
| 5e-4       | ~42%          | ~80%         |

> Actual numbers depend on hardware, CIFAR-10 download, and random seed.

---

## Design Decisions

| Decision | Rationale |
|:---------|:----------|
| Sigmoid activation on gate scores | Differentiable, bounded (0,1), gates are exactly zero-able |
| L1 (not L2) penalty on gates | Constant gradient pressure drives gates all the way to 0 |
| Gate initialised to sigmoid(1)≈0.73 | All connections start open; network learns what to prune |
| Bias not gated | Bias pruning rarely helps and adds instability |
| BatchNorm between layers | Stabilises training alongside the gate perturbation |
| Dropout(0.3) | Additional regularisation — prevents gates from doing all the work |

---

## Hyperparameter Tuning Tips

- **λ too low (< 1e-6)**: Model behaves like a standard dense network; minimal pruning.
- **λ too high (> 1e-3)**: Gates collapse too fast; training instability and poor accuracy.
- **Sweet spot**: 1e-5 to 5e-4 for this architecture on CIFAR-10.
- Try scheduling λ (start low, increase) for better accuracy-sparsity trade-off.
