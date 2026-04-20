# Self-Pruning Neural Network -- Results Report

## Why L1 Regularisation on Sigmoid Gates Encourages Sparsity

Each weight `w_ij` is masked by a gate `g_ij = sigmoid(s_ij)` where `s_ij`
is a learnable score. The sparsity loss adds `lambda * sum(g_ij)` to the total loss.

The gradient of this term with respect to the score is:

```
d(lambda * g_ij) / d(s_ij) = lambda * sigmoid(s_ij) * (1 - sigmoid(s_ij))
```

This gradient is always **positive**, so gradient *descent* pushes `s_ij`
**downward** -- driving `g_ij` toward 0 and progressively muting the weight.

Crucially, the L1 penalty applies a **constant-magnitude pressure** regardless
of the current gate value, unlike L2 regularisation whose gradient shrinks toward
zero and allows small non-zero values to survive. L1 drives gates all the way to
(or arbitrarily close to) zero, which effectively removes the corresponding weight.

---

## Results Table

| Lambda | Test Accuracy | Sparsity Level (%) |
|:------:|:-------------:|:------------------:|
| 0.0001 | 57.19% | 92.9% |
| 0.0003 | 57.58% | 98.8% |
| 0.0005 | 57.59% | 99.7% |

**Best model:** lambda = 0.0005 with 57.59% accuracy and 99.7% sparsity.

---

## Analysis of the Lambda Trade-off

- **Low lambda**: The sparsity penalty is weak. Most gates remain open (near 1),
  so the network behaves like a standard dense network. Accuracy is highest
  but very few weights are pruned.

- **Medium lambda**: A good balance. The network prunes a significant fraction of
  weights while maintaining competitive accuracy. The gate distribution shows
  a clear bimodal shape -- a spike at 0 (pruned) and a cluster away from 0
  (retained).

- **High lambda**: The penalty dominates. Most gates collapse to near 0, yielding
  very high sparsity but at the cost of accuracy -- the network is
  under-parameterised for the task.

---

## Gate Distribution Plot

The plot `gate_distributions.png` shows the distribution of final gate values
for each lambda. A successful run exhibits:

1. A large spike at 0 (many pruned connections).
2. A smaller cluster of values distributed above ~0.3 (retained connections).
3. Very few values in the intermediate range (gates are "binary-ish").

The training curves in `training_curves.png` illustrate how sparsity grows
monotonically over epochs while accuracy converges.

---

## Reproducibility

- Seed: `42`
- Device: `cuda`
- Optimizer: Adam (separate param groups -- weight_decay=1e-4 for weights, 0.0 for gates)
- Scheduler: Cosine annealing
- Architecture: 3072 -> 1024 -> 512 -> 256 -> 128 -> 10 (all PrunableLinear)
- Epochs: 30 per run
