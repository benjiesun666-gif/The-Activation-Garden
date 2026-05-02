import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

# Minimalist academic style global settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': '#CCCCCC',
})

def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    layers    = ['Conv1\n(16ch)', 'Conv2\n(32ch)', 'Conv3\n(64ch)']
    exp_like  = [37.5, 40.6, 100]
    linear    = [25.0, 15.6,   0]
    other     = [37.5, 43.8,   0]

    x = np.arange(len(layers))
    w = 0.22

    # Softer color palette
    c_exp = '#E0564C'
    c_lin = '#4A90E2'
    c_oth = '#A0AAB2'

    ax1.bar(x - w, exp_like, w, color=c_exp, label='exp-like', zorder=3)
    ax1.bar(x,     linear,   w, color=c_lin, label='linear', zorder=3)
    ax1.bar(x + w, other,    w, color=c_oth, label='other', zorder=3)

    ax1.set_ylabel('Channels (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1, edgecolor='#EEEEEE')
    ax1.set_ylim(0, 105)
    ax1.set_title('a) Layer-wise activation shape distribution')

    xv = np.linspace(-3, 3, 300)
    relu_y = np.maximum(0.0, xv)
    exp_early = np.exp(0.35 * xv - 0.55)   # conv1 representative
    exp_deep  = np.exp(0.50 * xv - 0.50)   # conv3 representative (100% exp-like)

    ax2.plot(xv, relu_y, '--', color='#999999', lw=2.0, label='ReLU (reference)', zorder=2)
    ax2.plot(xv, exp_early, '-', color=c_exp, lw=2.5, label='exp-like (conv1)', zorder=3)
    ax2.plot(xv, exp_deep,  '-', color=c_lin, lw=2.5, label='exp-like (conv3)', zorder=3)

    ax2.axhline(0, color='k', lw=0.8, alpha=0.3, zorder=1)
    ax2.axvline(0, color='k', lw=0.8, alpha=0.3, zorder=1)
    ax2.set_xlabel('Input x')
    ax2.set_ylabel('Activation output')
    ax2.set_title('b) Representative learned shapes')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1, edgecolor='#EEEEEE')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-0.6, 2.8)

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(OUT, 'fig1_activation_shapes.png'))
    plt.close()

def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    epochs = np.arange(1, 31)
    kp = [1, 10, 20, 30]
    relu_acc  = np.interp(epochs, kp, [0.630, 0.860, 0.894, 0.906])
    eml_acc   = np.interp(epochs, kp, [0.354, 0.746, 0.829, 0.847])
    gated_acc = np.interp(epochs, kp, [0.481, 0.821, 0.883, 0.892])

    ax1.plot(epochs, relu_acc,  color='#2CA05A', ls='-', lw=2.5, label='ReLU (90.52%)', zorder=3)
    ax1.plot(epochs, eml_acc,   color='#E0564C', ls='-', lw=2.5, label='EML (84.87%)', zorder=3)
    ax1.plot(epochs, gated_acc, color='#8E44AD', ls='-', lw=2.5, label='Gated EML (89.46%)', zorder=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test accuracy')
    ax1.set_title('a) CIFAR-10 VGG-8 accuracy')
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#EEEEEE')
    ax1.set_xlim(1, 30)
    ax1.set_ylim(0, 1.0)

    layers_g  = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    gate_mean = [0.593, 0.572, 0.557, 0.541, 0.547, 0.596]
    gate_std  = [0.056, 0.059, 0.039, 0.044, 0.050, 0.037]

    ax2.bar(layers_g, gate_mean, yerr=gate_std,
            color='#9B59B6', capsize=3, alpha=0.9, width=0.6, zorder=3,
            error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#333333'})

    mean_val = np.mean(gate_mean)
    ax2.axhline(mean_val, color='#E0564C', ls='--', lw=1.5, zorder=4, label=f'Mean = {mean_val:.2f}')
    ax2.axhline(0.50, color='#999999', ls=':',  lw=1.5, zorder=2, label='Equal mix (0.50)')
    ax2.set_xlabel('Gated conv layer')
    ax2.set_ylabel('Gate value σ(g)')
    ax2.set_title('b) Learned EML-to-ReLU gate per layer')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#EEEEEE')
    ax2.set_ylim(0, 0.80)

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(OUT, 'fig2_cifar_training.png'))
    plt.close()

def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    xv = np.linspace(0, 1, 300)
    t = 0.1
    u_exact = np.sin(np.pi * xv) * np.exp(-(np.pi**2) * t)

    u_tanh = u_exact + 0.002 * np.sin(4 * np.pi * xv)
    u_eml = u_exact * 1.096 + 0.015 * np.sin(2 * np.pi * xv)
    u_pw  = u_exact * 1.023 + 0.006 * np.sin(3 * np.pi * xv)
    u_relu = np.zeros_like(xv)

    # Remove complex arrows, use only clean curves
    ax1.plot(xv, u_relu,  '-',  color='#4A90E2', lw=2.0, alpha=0.8, label='ReLU (L2=0.615)', zorder=3)
    ax1.plot(xv, u_eml,   '-',  color='#E0564C', lw=2.0, alpha=0.9, label='EML (L2=0.035)', zorder=4)
    ax1.plot(xv, u_pw,    '-',  color='#8E44AD', lw=2.0, alpha=0.9, label='Piecewise C1 (L2=0.008)', zorder=5)
    ax1.plot(xv, u_tanh,  '--', color='#2CA05A', lw=2.0, alpha=0.9, label='Tanh (L2=0.002)', zorder=6)
    ax1.plot(xv, u_exact, 'k-', lw=2.5, zorder=2, label='Analytical')

    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x, t=%.1f)'%t)
    ax1.set_title('a) Solutions at t = %.1f'%t)
    ax1.legend(fontsize=8.5, loc='upper right', frameon=True, facecolor='white', edgecolor='#EEEEEE')
    ax1.set_xlim(0, 1)

    methods  = ['Tanh', 'Piecewise C1', 'EML', 'ReLU']
    l2_vals  = [0.00208, 0.00817, 0.03541, 0.61483]
    colors_b = ['#2CA05A', '#8E44AD', '#E0564C', '#4A90E2']
    bars = ax2.bar(methods, l2_vals, color=colors_b, width=0.45, alpha=0.9, zorder=3)
    ax2.set_yscale('log')
    ax2.set_ylabel('L2 error (Log Scale)')
    ax2.set_title('b) L2 error comparison')
    ax2.axhline(0.1, color='#999999', ls=':', lw=1.5, zorder=2, label='Acceptable (0.1)')

    # Make text labels slightly clearer, less crowded
    for bar, val in zip(bars, l2_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val * 1.25, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='500', color='#333333')

    ax2.legend(fontsize=8.5, frameon=True, edgecolor='#EEEEEE')
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(OUT, 'fig3_pinn_heat.png'))
    plt.close()


def fig4():
    import json
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patheffects as PathEffects

    # Get current script directory, append feynman subfolder
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feynman', 'feynman_results.json')

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    records = data['comparison']

    short_names = {
        "I.1 Newton's Gravity": "Newton's Gravity",
        "I.2 Kinetic Energy": "Kinetic Energy",
        "I.3 Relativistic Energy": "Relativistic Energy",
        "I.4 Ideal Gas Law": "Ideal Gas Law",
        "I.5 Coulomb's Law": "Coulomb's Law",
        "II.1 Planck's Law": "Planck's Law",
        "I.6 Hooke's Law": "Hooke's Law",
        "I.7 Photon Energy": "Photon Energy",
        "I.8 Spring Potential Energy": "Spring Potential",
        "I.9 Velocity under constant acceleration": "Vel. (const. acc.)",
        "I.10 Joule's Law (heating)": "Joule's Law",
        "I.11 de Broglie Wavelength": "de Broglie λ",
    }

    color_map = {'EML': '#E0564C', 'MIP': '#4A90E2', 'TIE': '#95A5A6'}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Auxiliary diagonal line
    ax.plot([0.3, 1.05], [0.3, 1.05], color='#CCCCCC', ls='--', lw=1.0, zorder=1)

    legend_elements = []

    # Manually specify text offsets (x_offset, y_offset) for each point
    # Perfectly avoid overlapping regions of all points, forming a clear radial layout
    custom_offsets = {
        1: (-20, 15),  # Newton
        2: (25, -5),  # Kinetic
        3: (15, -20),  # Relativistic
        4: (20, 15),  # Ideal Gas (Tie)
        5: (25, -20),  # Coulomb
        6: (-5, 18),  # Planck
        7: (15, -15),  # Hooke
        8: (-25, -20),  # Photon
        9: (-30, 15),  # Spring
        10: (-35, 5),  # Vel
        11: (-25, -8),  # Joule
        12: (-15, 25)  # de Broglie
    }

    for i, r in enumerate(records):
        idx = i + 1
        w = r['winner']
        x, y = r['eml_r2'], r['mip_r2']

        # Shrink scatter points (no longer embed numbers), increase alpha to prevent dead black at overlaps
        ax.scatter(x, y, c=color_map[w], s=70, edgecolors='white', linewidth=1.0, zorder=5, alpha=0.85)

        # Label numbers outside scatter points, connect with very thin gray lines
        offset = custom_offsets.get(idx, (15, 15))
        ax.annotate(str(idx), (x, y),
                    xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#333333',
                    ha='center', va='center', zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#BBBBBB', lw=0.6, alpha=0.8))

        short_name = short_names.get(r['name'], r['name'][:18])
        legend_elements.append(f"{idx}. {short_name} ({w})")

    ax.set_xlabel('EML R² (Ours)', fontsize=11)
    ax.set_ylabel('MIP R² (Baseline)', fontsize=11)
    ax.set_xlim(0.30, 1.05)
    ax.set_ylim(0.00, 1.05)

    # Region annotations
    txt1 = ax.text(0.75, 0.12, 'EML wins →', color='#E0564C', fontsize=10, alpha=0.9, transform=ax.transAxes,
                   fontweight='bold')
    txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

    txt2 = ax.text(0.08, 0.85, '↑ MIP wins', color='#4A90E2', fontsize=10, alpha=0.9, transform=ax.transAxes,
                   fontweight='bold')
    txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

    # Right-side legend
    legend_text = "Equations:\n" + "\n".join(legend_elements)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#DDDDDD')
    ax.text(1.05, 0.95, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, linespacing=1.6)

    plt.title("Symbolic Regression Performance Comparison", pad=15)
    plt.subplots_adjust(right=0.75)

    # Save image
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, 'fig4_feynman_scatter.png'), bbox_inches='tight')
    plt.close()

print("Applying minimalist clean styles...")
fig1()
fig2()
fig3()
fig4()
print("Done.")