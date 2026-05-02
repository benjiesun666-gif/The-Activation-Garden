# EML Experiments — Reproduction Guide

Paper: *"What You Activate Is What You Learn"*  
All paths relative to the `experiments/` directory.  
Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
cd experiments/
python cifar10/train_multiseed.py           # Core CIFAR experiment (~5 h)
python cifar10/train_multiseed_extension.py # CIFAR GELU/Swish/PReLU (~3 h)
python mnist/train_mnist.py                 # MNIST benchmark (~10 min)
python pinn/pinn_experiment.py              # PINN 1D heat equation (~5 min)
python pinn/pinn_extended.py                # 5-PDE benchmark (~30 min)
python feynman/feynman_experiment.py        # Feynman SR (~30 min)
python generate_figures.py                  # All figures (~10 sec)
```

## 1. MNIST (Table 1)

```bash
python mnist/train_mnist.py                # ReLU/GELU/Swish/EML accuracy (10 ep)
python mnist/train_mnist_multiseed.py      # Multi-seed activation shapes (5 seeds)
python mnist/train_mnist_reproduce.py      # Reproducible shape classification
```

Output: `mnist/results_mnist.json`, `multi_seed.json`, `activation_shapes.json`

## 2. CIFAR-10 VGG-8 (Table 2)

```bash
python cifar10/train_multiseed.py           # 5 methods x 4 seeds, 30 epochs each
python cifar10/train_multiseed_extension.py # GELU/Swish/PReLU x 3 seeds
```

Output: `cifar10/results_multiseed.json`, `results_multiseed_extension.json`

Failed experiments:
```bash
python cifar10/train_cifar10_hybrid.py      # Hybrid EML
python cifar10/train_cifar10_scaled.py      # Scaled EML
python cifar10/train_gated_resnet.py        # Gated EML on ResNet-18
```

## 3. PINN (Table 4, Table 5)

```bash
python pinn/pinn_experiment.py              # Heat: Tanh/ReLU/EML (5000 Adam)
python pinn/pinn_extended.py                # Burgers, Allen-Cahn, Beam, Lorenz
python pinn/pinn_pw.py                      # Piecewise C1 on 4 PDEs
python pinn/pinn_pwfree.py                  # Piecewise free on 4 PDEs
python pinn/beam_fix.py                     # Euler-Bernoulli with weighted losses
```

Output: `pinn/pinn_results.json`, `pinn_extended_results.json`, `pinn_pw_results.json`, `pinn_pwfree_results.json`

## 4. Feynman Symbolic Regression (Supp Table S1)

```bash
python feynman/feynman_experiment.py        # 12 equations, depth 3-4, 3000 epochs
```

Output: `feynman/feynman_results.json`

## 5. Supplementary Experiments

```bash
python variance_tracking.py                 # Fig. S1: per-layer variance (~10 sec)
python gate_sensitivity.py                  # Fig. S2: fixed gate sweep (~60 min)
python layernorm_test.py                    # BN vs IN vs NoNorm (~45 min)
python c1_continuity.py                     # C1 continuity analysis (~20 min)
python deep_convergence/deep_experiment.py  # Deep EML tree convergence (~30 min)
```

## 6. Figures

```bash
python generate_figures.py
# Output: figures/ (6 PNG files)
```

## 7. Audit Tools

```bash
python audit/strict_audit.py                # Data accuracy + method normalization
python audit/nmi_compliance_check.py        # Formatting compliance
python audit/final_audit.py                 # JSON existence + value checks
```

## Data File Map

| Paper Result | JSON File |
|-------------|-----------|
| MNIST accuracy | `mnist/results_mnist.json` |
| Activation shapes (Table 1) | `mnist/activation_shapes.json` |
| Multi-seed shape stats | `mnist/multi_seed.json` |
| CIFAR-10 main multi-seed (Table 2) | `cifar10/results_multiseed.json` |
| CIFAR-10 GELU/Swish/PReLU | `cifar10/results_multiseed_extension.json` |
| Piecewise C1 constrained | `cifar10/results_piecewise_c1_30ep.json` |
| PReLU baseline | `cifar10/results_prelu.json` |
| Gated ResNet | `cifar10/results_gated_resnet.json` |
| Gate sensitivity sweep | `cifar10/results_gate_sweep.json` |
| LayerNorm comparison | `cifar10/results_layernorm.json` |
| C1 continuity analysis | `cifar10/results_c1_continuity.json` |
| Hybrid EML (failed) | `cifar10/results_hybrid_eml.json` |
| Scaled EML (failed) | `cifar10/results_scaled_eml.json` |
| PINN Heat (Table 4) | `pinn/pinn_results.json` |
| PINN 5-PDE (Table 5) | `pinn/pinn_extended_results.json` |
| PINN Piecewise C1 | `pinn/pinn_pw_results.json` |
| PINN Piecewise Free | `pinn/pinn_pwfree_results.json` |
| Feynman 12eq R² | `feynman/feynman_results.json` |
| Deep convergence MSE | `deep_convergence/results_deep.json` |
