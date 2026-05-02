# -*- coding: utf-8 -*-
"""
Deep EML Experiments: Log-space vs standard, q-EML, depth-accuracy curves.

Tests:
  1. Log-space EML at depth 5, 6, 7 (where standard EML overflows)
  2. Standard EML at depth 2, 3, 4 as baseline
  3. q-EML with trainable q on same targets
  4. Depth-accuracy curves with inflection point analysis

Target functions:
  Gamma(1+x) via lgamma, erf(x), Airy Ai(x), sin(x)/x (sinc), J0(x) (Bessel J0)
"""
import torch, torch.nn as nn
import numpy as np, os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eml_trainable import TrainableEMLTreeV2
from logspace_eml import LogspaceEMLTreeV3, train_logspace, extract_formula_logspace
from q_eml import QEMLTree, train_q_eml
from direction_d_fitting import train_eml, safe_init, extract_formula
from scipy.special import airy as scipy_airy, j0 as scipy_j0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_d')
PLOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PLOTDIR, exist_ok=True)

SEED = 42
MAX_EPOCHS = 2000
N_PTS = 150

def get_targets(device, n_pts=200):
    targets = {}
    x_gamma = torch.linspace(0.5, 5.0, n_pts, device=device)
    targets['gamma'] = {'x': x_gamma, 'y': torch.lgamma(x_gamma), 'range': (0.5, 5.0), 'label': 'ln|Gamma(x)|'}

    x_erf = torch.linspace(-3.0, 3.0, n_pts, device=device)
    targets['erf'] = {'x': x_erf, 'y': torch.erf(x_erf), 'range': (-3.0, 3.0), 'label': 'erf(x)'}

    x_airy_np = np.linspace(-5.0, 5.0, n_pts)
    y_airy_np, _, _, _ = scipy_airy(x_airy_np)
    targets['airy'] = {'x': torch.from_numpy(x_airy_np).float().to(device),
                       'y': torch.from_numpy(y_airy_np).float().to(device),
                       'range': (-5.0, 5.0), 'label': 'Airy Ai(x)'}

    x_sinc = torch.linspace(-5.0, 5.0, n_pts, device=device)
    y_sinc = torch.where(x_sinc.abs() < 1e-8, torch.ones_like(x_sinc),
                         torch.sin(x_sinc) / x_sinc)
    targets['sinc'] = {'x': x_sinc, 'y': y_sinc, 'range': (-5.0, 5.0), 'label': 'sinc(x)'}

    x_j0 = torch.linspace(0.5, 10.0, n_pts)
    y_j0_np = scipy_j0(x_j0.numpy())
    targets['j0'] = {'x': x_j0.to(device),
                     'y': torch.from_numpy(y_j0_np).float().to(device),
                     'range': (0.5, 10.0), 'label': 'J0(x)'}

    return targets


def exp_standard_baseline(targets, device, depths=(2, 3, 4)):
    """Train standard EML at shallow depths."""
    print("\n" + "="*60)
    print("PART 1: Standard EML baseline (depths 2-4)")
    print("="*60)
    all_res = {}
    for name, tdata in targets.items():
        print(f"\n--- {tdata['label']} ---")
        res = {'func': name}
        for d in depths:
            torch.manual_seed(SEED)
            tree = TrainableEMLTreeV2(depth=d)
            safe_init(tree, d)
            r = train_eml(tree, tdata['x'], tdata['y'], n_epochs=MAX_EPOCHS,
                          lr=0.01, device=device, patience=1200, vstep=1000)
            fi = extract_formula(r['sd'], d) if r['sd'] else None
            res[d] = {**r, 'finfo': fi}
            status = f"MSE={r['mse']:.4e}" if r['ok'] else f"FAILED({r.get('mse','?')})"
            print(f"  depth={d}: {status}")
        all_res[name + '_std'] = res
    return all_res


def exp_logspace_deep(targets, device, depths=(5, 6)):
    """Train log-space EML at deep depths."""
    print("\n" + "="*60)
    print("PART 2: Log-space EML (depths 5-6)")
    print("="*60)
    all_res = {}
    for name, tdata in targets.items():
        print(f"\n--- {tdata['label']} ---")
        res = {'func': name}
        for d in depths:
            torch.manual_seed(SEED)
            tree = LogspaceEMLTreeV3(depth=d)
            tree.init_safe()
            r = train_logspace(tree, lambda xo: tdata['y'],
                               x_data=tdata['x'], y_data=tdata['y'],
                               n_epochs=MAX_EPOCHS, lr=0.01, device=device,
                               patience=1200, verbose=True, vstep=1000)
            fi = extract_formula_logspace(r['sd'], d) if r['sd'] else None
            res[d] = {**r, 'finfo': fi}
            status = f"MSE={r['mse']:.4e}" if r['ok'] else f"FAILED"
            print(f"  depth={d}: {status}")
        all_res[name + '_log'] = res
    return all_res


def exp_qeml(targets, device, depth=3, q_fixed=(0.5, 1.0, 1.3)):
    """Train q-EML at multiple fixed q values and with trainable q."""
    print("\n" + "="*60)
    print("PART 3: q-EML experiments (depth=3)")
    print("="*60)
    all_res = {}
    for name, tdata in targets.items():
        print(f"\n--- {tdata['label']} ---")
        res = {'func': name, 'fixed_q': {}, 'trainable_q': None}

        for qv in q_fixed:
            torch.manual_seed(SEED)
            tree = QEMLTree(depth=depth, q_init=qv)
            tree.init_safe()
            r = train_q_eml(tree, tdata['x'], tdata['y'], n_epochs=MAX_EPOCHS,
                            lr=0.01, device=device, patience=1200, vstep=1000)
            res['fixed_q'][qv] = r
            if r['ok']:
                print(f"  q_fixed={qv:.1f}: MSE={r['mse']:.4e}  q_final={r.get('q_final',0):.4f}")

        torch.manual_seed(SEED)
        tree = QEMLTree(depth=depth, q_init=1.0)
        tree.init_safe()
        r = train_q_eml(tree, tdata['x'], tdata['y'], n_epochs=MAX_EPOCHS,
                        lr=0.01, device=device, patience=1200, vstep=1000)
        res['trainable_q'] = r
        if r['ok']:
            print(f"  trainable_q: MSE={r['mse']:.4e}  q_final={r.get('q_final',0):.4f}")
        all_res[name] = res
    return all_res


def build_summary(all_std, all_log, all_q):
    summary = []
    for func_name in ['gamma', 'erf', 'airy', 'sinc', 'j0']:
        entry = {'func': func_name}

        key_std = func_name + '_std'
        if key_std in all_std:
            for d in [2, 3, 4]:
                r = all_std[key_std].get(d, {})
                if r.get('ok'):
                    entry[f'std_d{d}_mse'] = r['mse']
                    entry[f'std_d{d}_mae'] = r.get('mae', float('nan'))

        key_log = func_name + '_log'
        if key_log in all_log:
            for d in [5, 6]:
                r = all_log[key_log].get(d, {})
                if r.get('ok'):
                    entry[f'log_d{d}_mse'] = r['mse']
                    entry[f'log_d{d}_mae'] = r.get('mae', float('nan'))

        if func_name in all_q:
            tr = all_q[func_name].get('trainable_q', {})
            if tr.get('ok'):
                entry['q_trainable_mse'] = tr['mse']
                entry['q_final'] = tr.get('q_final', float('nan'))
                entry['q_mse'] = tr['mse']

        summary.append(entry)
    return summary


def plot_depth_accuracy(summary, savepath):
    func_names_plot = ['gamma', 'erf', 'airy', 'sinc', 'j0']
    labels = ['ln|Gamma(x)|', 'erf(x)', 'Airy Ai(x)', 'sinc(x)', 'J0(x)']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    fig, ax = plt.subplots(figsize=(12, 7))
    marker_std = 'o'
    marker_log = 's'

    for fi, (fn, label, color) in enumerate(zip(func_names_plot, labels, colors)):
        depths_std = []
        mses_std = []
        depths_log = []
        mses_log = []
        for s in summary:
            if s['func'] != fn:
                continue
            for d in [2, 3, 4]:
                key = f'std_d{d}_mse'
                if key in s and np.isfinite(s[key]):
                    depths_std.append(d)
                    mses_std.append(s[key])
            for d in [5, 6]:
                key = f'log_d{d}_mse'
                if key in s and np.isfinite(s[key]):
                    depths_log.append(d)
                    mses_log.append(s[key])

        if depths_std:
            ax.semilogy(depths_std, mses_std, marker=marker_std, color=color,
                        ls='-', lw=2, label=label)
        if depths_log:
            ax.semilogy(depths_log, mses_log, marker=marker_log, color=color,
                        ls='--', lw=2, alpha=0.7)

    ax.set_xlabel('Depth', fontsize=13)
    ax.set_ylabel('MSE (log scale)', fontsize=13)
    ax.set_title('Depth vs Accuracy: Standard depth 2-4 + Log-space depth 5-6', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xticks([2, 3, 4, 5, 6])
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"  plot -> {savepath}")


def plot_q_convergence(all_q, savepath):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    func_labels = {
        'gamma': ('ln|Gamma(x)|', '#3498db'),
        'erf': ('erf(x)', '#e74c3c'),
        'airy': ('Airy Ai(x)', '#2ecc71'),
        'sinc': ('sinc(x)', '#9b59b6'),
        'j0': ('J0(x)', '#f39c12'),
    }

    for fi, (fn, (label, color)) in enumerate(func_labels.items()):
        ax = axes[fi]
        qres = all_q.get(fn, {})
        tr = qres.get('trainable_q', {})
        q_hist = tr.get('q_hist', [])
        if q_hist:
            epochs = [e for e, _ in q_hist]
            q_vals = [v for _, v in q_hist]
            ax.plot(epochs, q_vals, color=color, lw=2)
            ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='q=1 (classical)')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('q')
        ax.grid(alpha=0.3)

    axes[5].axis('off')
    plt.suptitle('q-EML: q Parameter Convergence During Training', fontsize=15)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"  plot -> {savepath}")


def plot_fit_comparison(targets, all_std, all_log, savepath):
    func_names_plot = ['gamma', 'erf', 'airy', 'sinc', 'j0']
    labels = ['ln|Gamma(x)|', 'erf(x)', 'Airy Ai(x)', 'sinc(x)', 'J0(x)']

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for fi, (fn, label) in enumerate(zip(func_names_plot, labels)):
        ax = axes[fi]
        tdata = targets[fn]
        x_np = tdata['x'].cpu().numpy()
        y_np = tdata['y'].cpu().numpy()
        ax.plot(x_np, y_np, 'k-', lw=2, label='True')

        for depth, sty, name in [(2, '--', 'std d2'), (4, '-.', 'std d4'), (5, ':', 'log d5')]:
            for all_r, prefix, model_cls in [(all_std, '_std', TrainableEMLTreeV2),
                                              (all_log, '_log', None)]:
                key = fn + prefix
                if key not in all_r:
                    continue
                r = all_r[key].get(depth)
                if not r or not r.get('ok') or r.get('sd') is None:
                    continue
                if model_cls is TrainableEMLTreeV2:
                    m = model_cls(depth=depth)
                    m.load_state_dict(r['sd'])
                    m.eval()
                    with torch.no_grad():
                        yp = m(torch.from_numpy(x_np).float().to(tdata['x'].device)).cpu().numpy()
                else:
                    m = LogspaceEMLTreeV3(depth=depth)
                    m.load_state_dict(r['sd'])
                    m.eval()
                    with torch.no_grad():
                        yp = m(torch.from_numpy(x_np).float().to(tdata['x'].device)).cpu().numpy()
                ax.plot(x_np, yp, sty, lw=1.2, alpha=0.7,
                        label=f'{name} MSE={r["mse"]:.2e}')

        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    axes[5].axis('off')
    plt.suptitle('EML Fit Comparison: Standard vs Log-space', fontsize=15)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"  plot -> {savepath}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    t0 = time.time()

    targets = get_targets(device, n_pts=N_PTS)

    all_std = exp_standard_baseline(targets, device, depths=(2, 3, 4))
    all_log = exp_logspace_deep(targets, device, depths=(5, 6))
    all_q = exp_qeml(targets, device, depth=3, q_fixed=())

    summary = build_summary(all_std, all_log, all_q)
    summary_table = "\n".join(
        f"{s['func']:>8}: " + "  ".join(
            f"{k}={s[k]:.3e}" if isinstance(s.get(k), float) else f"{k}={s.get(k)}"
            for k in sorted(s.keys()) if k != 'func'
        ) for s in summary
    )
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary_table)

    results = {
        'summary_table': str(summary_table),
        'summary_json': summary,
    }

    torch.save({
        'std': all_std,
        'log': all_log,
        'q': all_q,
        'summary': summary,
        'summary_table': str(summary_table)
    },
    os.path.join(OUTDIR, 'deep_experiment_full.pt'))
    print(f"\nFull results saved to {os.path.join(OUTDIR, 'deep_experiment_full.pt')}")

    plot_depth_accuracy(summary, os.path.join(PLOTDIR, 'depth_accuracy.png'))
    plot_q_convergence(all_q, os.path.join(PLOTDIR, 'q_convergence.png'))
    plot_fit_comparison(targets, all_std, all_log, os.path.join(PLOTDIR, 'fit_comparison.png'))

    elapsed = time.time() - t0
    print(f"\nAll experiments done. Total time: {elapsed/60:.1f} min")
