"""
Bullet #1: Variance tracking through VGG-8 forward pass.
Compare ReLU vs EML vs Gated EML: per-layer feature map variance.
No training needed — pure initialization forward pass.
"""
import torch, torch.nn as nn, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..')); from eml_dl import EMLActivation

device='cpu'

# Build VGG-8: 6 conv blocks + 3 pools = 6 activation points
def build_vgg(act_type):
    layers = []
    ch = [(3,64), (64,64), (64,128), (128,128), (128,256), (256,256)]
    pool_after = {1:True, 3:True, 5:True}
    for i,(cin,cout) in enumerate(ch):
        layers.append(nn.Conv2d(cin,cout,3,padding=1,bias=False))
        layers.append(nn.BatchNorm2d(cout))
        if act_type=='relu': layers.append(nn.ReLU())
        elif act_type=='eml': layers.append(EMLActivation(cout,init='identity'))
        elif act_type=='gated': 
            class G(nn.Module):
                def __init__(self): super().__init__(); self.eml=EMLActivation(cout,init='identity'); self.relu=nn.ReLU()
                def forward(self,x): return 0.57*self.eml(x)+0.43*self.relu(x)
            layers.append(G())
        if i in pool_after: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# Run forward pass and track variance after each activation
def track_variance(model, name):
    model.eval()
    x = torch.randn(64, 3, 32, 32)  # CIFAR-sized random input
    variances = []
    layer_idx = 0
    with torch.no_grad():
        for module in model:
            x = module(x)
            # Record after every activation (Conv-BN-Act → record after Act)
            if isinstance(module, (nn.ReLU, EMLActivation)) or 'G' in str(type(module)):
                variances.append(float(torch.var(x).item()))
                layer_idx += 1
    return variances

print('Tracking variance through VGG-8 forward pass...')
var_relu = track_variance(build_vgg('relu'), 'ReLU')
var_eml = track_variance(build_vgg('eml'), 'EML')
var_gated = track_variance(build_vgg('gated'), 'Gated')

layers = [f'L{i+1}' for i in range(len(var_relu))]
relu_str = [v for v in var_relu]; eml_str = [v for v in var_eml]; gated_str = [v for v in var_gated]
print('  ReLU:  ' + str([round(v,4) for v in var_relu]))
print('  EML:   ' + str([round(v,4) for v in var_eml]))
print('  Gated: ' + str([round(v,4) for v in var_gated]))

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1,7), var_relu, 'o-', color='#2CA05A', lw=2.5, markersize=8, label='ReLU')
ax.plot(range(1,7), var_eml, 's-', color='#E0564C', lw=2.5, markersize=8, label='EML (pure)')
ax.plot(range(1,7), var_gated, 'D-', color='#8E44AD', lw=2.5, markersize=8, label='Gated EML')
ax.set_yscale('log')
ax.set_xlabel('Convolutional block (depth)')
ax.set_ylabel('Feature map variance (log scale)')
ax.set_title('Variance propagation through VGG-8 layers (forward pass)')
ax.legend(frameon=True, facecolor='white', edgecolor='#EEEEEE')
ax.set_xticks(range(1,7))
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'fig_variance_propagation.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_variance_propagation.png')

# Save results as JSON
import json
out = {'experiment': 'Variance propagation through VGG-8',
       'method': 'forward pass with random Gaussian input',
       'results': {
           'ReLU': {'L1': round(var_relu[0],4), 'L6': round(var_relu[-1],4), 'all': [round(v,4) for v in var_relu]},
           'EML':  {'L1': round(var_eml[0],4),  'L6': round(var_eml[-1],4),  'all': [round(v,4) for v in var_eml]},
           'Gated':{'L1': round(var_gated[0],4),'L6': round(var_gated[-1],4),'all': [round(v,4) for v in var_gated]}},
       'script': 'variance_tracking.py', 'date': '2026-04-30'}
with open(os.path.join(os.path.dirname(__file__), 'results_variance.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('Saved: results_variance.json')
