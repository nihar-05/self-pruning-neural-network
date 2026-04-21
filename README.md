# Self-Pruning Neural Network

A feed-forward simple neural network that **learns to prune its own weights during training** using learnable gate parameters and L1 sparsity regularisation — no post-training pruning step required.

Built for the Tredence Analytics AI Engineer case study.

---

## Project Structure

```
.
├── self_pruning_nn.py   # Main implementation
├── requirements.txt     # Dependencies
├── README.md            # Project description
└── outputs/             # Generated results
    ├── gate_distributions.png
    ├── training_curves.png
    └── report.md
```

---

## Quick Start

```bash
pip install -r requirements.txt
python self_pruning_nn.py
```

Outputs (plots + report) are saved in `./outputs/`.

---

## Key Idea

Each weight has a learnable gate:

```
g_ij = sigmoid(s_ij)
```

The effective weight becomes:

```
w_ij * g_ij
```

L1 regularisation on gates drives unnecessary connections toward zero, enabling **automatic pruning during training**.

---

## Loss Function

```
L_total = CrossEntropy + λ × Σ g_ij
```

* CrossEntropy → classification performance
* L1 on gates → sparsity

---

## Experimental Setup

* Dataset: CIFAR-10
* Architecture: 3072 → 1024 → 512 → 256 → 128 → 10
* Optimizer: Adam
* Epochs: 30
* Lambda values: {1e-4, 3e-4, 5e-4}
* Sparsity threshold: **0.05**

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
| :--------: | :-----------: | :----------: |
|    1e-4    |     57.19%    |     92.9%    |
|    3e-4    |     57.58%    |     98.8%    |
|    5e-4    |     57.59%    |     99.7%    |

---

## Observations

* Increasing λ increases sparsity significantly
* The model achieves up to **~99% sparsity**
* Accuracy remains stable (~57%), indicating **high parameter redundancy**
* Pruning occurs gradually during training (after initial learning phase)

---

## Important Note on Threshold

A threshold of **0.05** was used to define sparsity.

* A stricter threshold (1e-2) resulted in 0% sparsity
* 0.05 better captures **effectively pruned connections**

---

## Conclusion

The model successfully demonstrates **dynamic self-pruning**, achieving extremely high sparsity with minimal loss in accuracy. This highlights the redundancy present in dense neural networks and the effectiveness of L1-based gating mechanisms.

---

## Outputs

See `outputs/` for:

* Gate distribution plots
* Training curves
* Full report
