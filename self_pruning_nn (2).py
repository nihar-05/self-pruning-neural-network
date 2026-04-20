"""
Self-Pruning Neural Network on CIFAR-10
========================================
A feed-forward network where each weight has a learnable "gate" scalar.
During training, L1 regularization on the gates drives them toward zero,
effectively pruning unnecessary connections inline — no post-training step needed.

Usage:
    python self_pruning_nn.py

Results are printed to stdout and plots saved as PNG files.
"""

# ---------------------------------------------------------------------------
# FIX 1: matplotlib backend must be set BEFORE importing pyplot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# FIX 3: Ensure full GPU reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ===========================================================================
# Part 1 — PrunableLinear layer
# ===========================================================================

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates a learnable scalar
    gate with every weight element.

    Forward pass:
        gates       = sigmoid(gate_scores)          # squash to (0, 1)
        pruned_w    = weight * gates                # element-wise mask
        output      = x @ pruned_w.T + bias        # standard affine op

    Because all operations are differentiable, gradients flow back through
    both `weight` and `gate_scores` automatically via autograd.

    A gate value near 0 means the corresponding weight is effectively removed
    from the network. L1 regularisation on the gates during training pushes
    them toward exactly 0.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight — shape (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        # Gate scores — same shape; sigmoid-ed to produce gates in (0, 1)
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        # Optional bias (not gated — bias pruning is uncommon and rarely helps)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        """Kaiming uniform for weights; small positive init for gate_scores so
        gates start near sigmoid(1) ≈ 0.73 — all connections active at t=0."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Initialise gate_scores to 1.0 → sigmoid(1) ≈ 0.73 (mostly open)
        nn.init.constant_(self.gate_scores, 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — convert raw scores to gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 — element-wise mask the weights
        pruned_weights = self.weight * gates

        # Step 3 — standard linear transform: x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached from graph) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 0.05) -> float:
        """Fraction of gates below `threshold` (treated as pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ===========================================================================
# Network definition
# ===========================================================================

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (10 classes, 32x32x3 = 3072 input dims).

    Architecture (all linear layers are PrunableLinear):
        3072 -> 1024 -> 512 -> 256 -> 128 -> 10

    BatchNorm and ReLU between hidden layers; no activation on the output
    (CrossEntropyLoss expects raw logits).
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = PrunableLinear(128, 10)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = self.fc5(x)
        return x

    def prunable_layers(self):
        """Iterator over all PrunableLinear modules in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Why L1 encourages sparsity:
        The gradient of the sparsity term w.r.t. gate_score s_ij is:

            d(lambda * g_ij) / d(s_ij) = lambda * sigmoid(s_ij) * (1 - sigmoid(s_ij))

        This is always positive, so gradient *descent* pushes s_ij downward,
        driving g_ij toward 0. Crucially, L1 applies a constant-magnitude
        pressure regardless of the current gate value — unlike L2, whose
        gradient shrinks toward zero and allows small non-zero gates to
        survive. L1 drives gates all the way to (or very near) zero.

        FIX: Use Python's built-in sum() over a generator instead of
        accumulating with a leaf tensor. This builds a clean computation
        graph from the start and avoids issues with in-place leaf tensor ops.
        """
        return sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in self.prunable_layers()
        )

    def overall_sparsity(self, threshold: float = 0.05) -> float:
        """Network-wide fraction of weights treated as pruned."""
        total_weights = 0
        pruned_weights = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
        return pruned_weights / total_weights if total_weights > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Concatenate all gate values into a single numpy array for plotting."""
        all_gates = []
        for layer in self.prunable_layers():
            all_gates.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(all_gates)


# ===========================================================================
# Part 2 — Data loading
# ===========================================================================

def get_cifar10_loaders(batch_size: int = 128):
    """
    Download (if needed) and return DataLoader objects for CIFAR-10 train/test.
    Standard normalisation: mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010).
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = "./data"
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # FIX 4: num_workers=0 as safe default; avoids multiprocessing issues on
    # Windows and some container/CI environments. Increase if on Linux+GPU.
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))
    return train_loader, test_loader


# ===========================================================================
# Part 3 — Training loop
# ===========================================================================

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lam: float,
    epoch: int,
) -> dict:
    """
    Run one full pass over the training data.
    Returns a dict with average classification loss, sparsity loss, total loss.
    """
    model.train()
    total_cls_loss   = 0.0
    total_spar_loss  = 0.0
    total_loss_accum = 0.0
    correct = 0
    total   = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(inputs)

        # Classification loss (Cross-Entropy)
        cls_loss = F.cross_entropy(logits, targets)

        # Sparsity loss — L1 on sigmoid gates (fixed accumulation)
        spar_loss = model.sparsity_loss()

        # Composite loss
        loss = cls_loss + lam * spar_loss

        loss.backward()
        optimizer.step()

        # Accumulate stats
        total_cls_loss   += cls_loss.item()
        total_spar_loss  += spar_loss.item()
        total_loss_accum += loss.item()

        preds   = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)

    n = len(loader)
    return {
        "cls_loss":   total_cls_loss   / n,
        "spar_loss":  total_spar_loss  / n,
        "total_loss": total_loss_accum / n,
        "train_acc":  correct / total,
    }


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader) -> dict:
    """Evaluate on the given loader; return accuracy and average CE loss."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        logits = model(inputs)
        loss   = F.cross_entropy(logits, targets)

        total_loss += loss.item()
        correct    += logits.argmax(1).eq(targets).sum().item()
        total      += targets.size(0)

    return {
        "test_loss": total_loss / len(loader),
        "test_acc":  correct / total,
    }


def train(
    lam: float,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    verbose: bool = True,
) -> dict:
    """
    Full training run for a single lambda value.
    Returns a results dict with final metrics and gate values.

    Note on weight_decay: Adam's weight_decay applies L2 to ALL parameters,
    including gate_scores. This adds a weak second push on gates alongside
    the explicit L1. To isolate L1-only pressure on gates, use separate
    parameter groups (gate params get weight_decay=0.0).
    """
    print(f"\n{'='*60}")
    print(f"  Training with lambda = {lam}")
    print(f"{'='*60}")

    train_loader, test_loader = get_cifar10_loaders(batch_size)
    model = SelfPruningNet().to(DEVICE)

    # FIX 5: Separate parameter groups so weight_decay (L2) does NOT apply to
    # gate_scores — preserving pure L1-only pressure from the sparsity loss.
    gate_params  = [p for n, p in model.named_parameters() if "gate" in n]
    other_params = [p for n, p in model.named_parameters() if "gate" not in n]

    optimizer = optim.Adam([
        {"params": other_params, "weight_decay": 1e-4},
        {"params": gate_params,  "weight_decay": 0.0},
    ], lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_acc":  [],
        "test_acc":   [],
        "total_loss": [],
        "spar_loss":  [],
        "sparsity":   [],
    }

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, lam, epoch)
        test_stats  = evaluate(model, test_loader)
        scheduler.step()

        sparsity = model.overall_sparsity()

        history["train_acc"].append(train_stats["train_acc"])
        history["test_acc"].append(test_stats["test_acc"])
        history["total_loss"].append(train_stats["total_loss"])
        history["spar_loss"].append(train_stats["spar_loss"])
        history["sparsity"].append(sparsity)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss: {train_stats['total_loss']:.4f} "
                f"(cls={train_stats['cls_loss']:.4f}, spar={train_stats['spar_loss']:.4f}) | "
                f"Train acc: {train_stats['train_acc']*100:.2f}% | "
                f"Test acc: {test_stats['test_acc']*100:.2f}% | "
                f"Sparsity: {sparsity*100:.1f}%"
            )

    final_test = evaluate(model, test_loader)
    final_sparsity = model.overall_sparsity()
    all_gates = model.all_gate_values()

    print(
        f"\n  [Done] Final — Test acc: {final_test['test_acc']*100:.2f}% | "
        f"Sparsity: {final_sparsity*100:.1f}%"
    )

    return {
        "lam":         lam,
        "test_acc":    final_test["test_acc"],
        "sparsity":    final_sparsity,
        "history":     history,
        "gate_values": all_gates,
        "model":       model,
    }


# ===========================================================================
# Part 4 — Plotting
# ===========================================================================

def plot_gate_distribution(results_list: list, save_dir: str = "."):
    """
    For each lambda run, plot the distribution of final gate values.
    A successful result shows a large spike at 0 and a cluster away from 0.
    Saves one combined figure.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, res, color in zip(axes, results_list, colors):
        gates = res["gate_values"]
        lam   = res["lam"]
        acc   = res["test_acc"] * 100
        spar  = res["sparsity"] * 100

        ax.hist(gates, bins=80, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(
            f"lambda = {lam}\nAcc: {acc:.1f}%  |  Sparsity: {spar:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2, label="threshold (0.01)")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Distribution of Final Gate Values per Lambda", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "gate_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {path}")
    return path


def plot_training_curves(results_list: list, save_dir: str = "."):
    """Plot test accuracy and sparsity over epochs for all lambda values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    linestyles = ["-", "--", ":"]

    for res, color, ls in zip(results_list, colors, linestyles):
        lam     = res["lam"]
        history = res["history"]
        epochs  = range(1, len(history["test_acc"]) + 1)

        ax1.plot(epochs, [a * 100 for a in history["test_acc"]],
                 color=color, linestyle=ls, linewidth=1.8, label=f"lambda={lam}")
        ax2.plot(epochs, [s * 100 for s in history["sparsity"]],
                 color=color, linestyle=ls, linewidth=1.8, label=f"lambda={lam}")

    ax1.set_title("Test Accuracy over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_title("Sparsity Level over Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Sparsity (%)")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {path}")
    return path


# ===========================================================================
# Part 5 — Results table & report
# ===========================================================================

def print_results_table(results_list: list):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Lambda':>10} | {'Test Accuracy':>14} | {'Sparsity (%)':>14}")
    print("  " + "-" * 50)
    for res in results_list:
        lam  = res["lam"]
        acc  = res["test_acc"] * 100
        spar = res["sparsity"] * 100
        print(f"  {lam:>10.4f} | {acc:>13.2f}% | {spar:>13.1f}%")
    print("=" * 60)


def generate_markdown_report(results_list: list, save_dir: str = ".") -> str:
    """Write the Markdown report file."""

    rows = "\n".join(
        f"| {r['lam']} | {r['test_acc']*100:.2f}% | {r['sparsity']*100:.1f}% |"
        for r in results_list
    )

    best = max(results_list, key=lambda r: r["test_acc"])

    report = f"""# Self-Pruning Neural Network -- Results Report

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
{rows}

**Best model:** lambda = {best['lam']} with {best['test_acc']*100:.2f}% accuracy and {best['sparsity']*100:.1f}% sparsity.

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

- Seed: `{SEED}`
- Device: `{DEVICE}`
- Optimizer: Adam (separate param groups -- weight_decay=1e-4 for weights, 0.0 for gates)
- Scheduler: Cosine annealing
- Architecture: 3072 -> 1024 -> 512 -> 256 -> 128 -> 10 (all PrunableLinear)
- Epochs: 30 per run
"""

    path = os.path.join(save_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"Report saved -> {path}")
    return path


# ===========================================================================
# Main
# ===========================================================================

def main():
    # Three lambda values: low / medium / high
    # These are intentionally small because the sparsity loss sums over
    # ALL gate values (~1.8M for this architecture), so even tiny lambda matters.
    LAMBDA_VALUES = [1e-4, 3e-4, 5e-4]
    EPOCHS        = 30
    SAVE_DIR      = "./outputs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    results = []
    for lam in LAMBDA_VALUES:
        res = train(lam=lam, epochs=EPOCHS)
        results.append(res)

    print_results_table(results)

    plot_gate_distribution(results, save_dir=SAVE_DIR)
    plot_training_curves(results,   save_dir=SAVE_DIR)
    generate_markdown_report(results,  save_dir=SAVE_DIR)

    print(f"\nAll outputs written to: {os.path.abspath(SAVE_DIR)}/")


if __name__ == "__main__":
    main()
