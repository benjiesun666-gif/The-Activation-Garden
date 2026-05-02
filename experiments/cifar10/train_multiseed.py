"""
Multi-seed reproducibility: exact same code, same epochs, different seeds.
Runs: ReLU, EML, Gated EML, Fixed Gate, Piecewise EML on CIFAR-10 VGG-8.
3 additional seeds per method (total 4 with existing run).
Uses EXACT architecture/training from original scripts.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time, numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda'
B, E, LR = 128, 30, 0.001

# Dataset (same as all original CIFAR scripts)
tform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
tform_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

def get_loaders():
    train_ds = datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'), train=True, download=False, transform=tform)
    test_ds = datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'), train=False, download=False, transform=tform_t)
    return (DataLoader(train_ds, B, shuffle=True, num_workers=0),
            DataLoader(test_ds, B, shuffle=False, num_workers=0))

# ===========================================================================
# Model builders — EXACT copies from original scripts
# ===========================================================================

def conv_block_relu(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
def conv_block_eml(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), EMLActivation(out_c, init='identity'))

class ReLUVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            conv_block_relu(3,64), conv_block_relu(64,64), nn.MaxPool2d(2),
            conv_block_relu(64,128), conv_block_relu(128,128), nn.MaxPool2d(2),
            conv_block_relu(128,256), conv_block_relu(256,256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256,10)
    def forward(self, x): return self.head(self.body(x).flatten(1))

class EMLVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            conv_block_eml(3,64), conv_block_eml(64,64), nn.MaxPool2d(2),
            conv_block_eml(64,128), conv_block_eml(128,128), nn.MaxPool2d(2),
            conv_block_eml(128,256), conv_block_eml(256,256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256,10)
    def forward(self, x): return self.head(self.body(x).flatten(1))

class GatedEML(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.eml = EMLActivation(ch, init='identity')
        self.relu = nn.ReLU(inplace=False)
        self.gate = nn.Parameter(torch.full((ch,), 0.5))
    def forward(self, x):
        g = torch.sigmoid(self.gate).view(1,-1,1,1)
        return g * self.eml(x) + (1-g) * self.relu(x)

def conv_block_gated(i,o):
    return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), GatedEML(o))

class GatedVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Sequential(
            conv_block_gated(3,64), conv_block_gated(64,64), nn.MaxPool2d(2),
            conv_block_gated(64,128), conv_block_gated(128,128), nn.MaxPool2d(2),
            conv_block_gated(128,256), conv_block_gated(256,256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Linear(256,10)
    def forward(self, x): return self.h(self.b(x).flatten(1))

class FixedGateEML(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.eml = EMLActivation(ch, init='identity')
        self.relu = nn.ReLU(inplace=False)
        self.gv = 0.57
    def forward(self, x):
        return self.gv * self.eml(x) + (1-self.gv) * self.relu(x)

def conv_block_fixed(i,o):
    return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), FixedGateEML(o))

class FixedVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Sequential(
            conv_block_fixed(3,64), conv_block_fixed(64,64), nn.MaxPool2d(2),
            conv_block_fixed(64,128), conv_block_fixed(128,128), nn.MaxPool2d(2),
            conv_block_fixed(128,256), conv_block_fixed(256,256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Linear(256,10)
    def forward(self, x): return self.h(self.b(x).flatten(1))

class PiecewiseFree(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Parameter(torch.full((ch,), -0.7))
        self.b = nn.Parameter(torch.full((ch,), 0.3))
        self.c = nn.Parameter(torch.full((ch,), 1.0))
        self.d = nn.Parameter(torch.full((ch,), -0.1))
        self.alpha = nn.Parameter(torch.full((ch,), 0.1))
        self.beta = nn.Parameter(torch.full((ch,), 0.5))
        self.gamma = nn.Parameter(torch.full((ch,), 0.5))
    def forward(self, x):
        a, b, c, d, alpha, beta, gamma = self.a, self.b, self.c, self.d, self.alpha, self.beta, self.gamma
        if x.dim() == 4:
            a, b, c, d, alpha, beta, gamma = [p.view(1,-1,1,1) for p in [a,b,c,d,alpha,beta,gamma]]
        left = b*x + a; right = d*x + c
        eml_out = torch.exp(torch.clamp(left, max=15.)) - torch.log(torch.clamp(right, min=1e-8))
        pos_out = alpha * x**2 + beta * x + gamma
        out = torch.where(x <= 0, eml_out, pos_out)
        return torch.clamp(out, -10., 10.)

def conv_block_pw(i,o):
    return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), PiecewiseFree(o))

class PW_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Sequential(
            conv_block_pw(3,64), conv_block_pw(64,64), nn.MaxPool2d(2),
            conv_block_pw(64,128), conv_block_pw(128,128), nn.MaxPool2d(2),
            conv_block_pw(128,256), conv_block_pw(256,256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Linear(256,10)
    def forward(self, x): return self.h(self.b(x).flatten(1))

# ===========================================================================
# Training — EXACT same config as all original CIFAR scripts
# ===========================================================================

def train_one(model, name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = get_loaders()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best = 0; t0 = time.time()
    print(f'\n  [{name}] seed={seed} | {n:,} params')
    for ep in range(1, E+1):
        model.train(); tl, tc, tn = 0., 0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            loss = F.cross_entropy(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item()*x.size(0); tc += (model(x).argmax(1)==y).sum().item(); tn += x.size(0)
        model.eval(); vl, vc, vn = 0., 0, 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(device), y.to(device); o = model(x)
                vl += F.cross_entropy(o, y).item()*x.size(0); vc += (o.argmax(1)==y).sum().item(); vn += x.size(0)
        sched.step(); best = max(best, vc/vn)
        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f'    ep{ep:2d}: acc={vc/vn:.4f} best={best:.4f} | {elapsed:.0f}s')
    return best

# ===========================================================================
# Run: 4 seeds per method (seed 0,1,2,3)
# ===========================================================================

methods = [
    ('ReLU', ReLUVGG),
    ('EML', EMLVGG),
    ('Gated', GatedVGG),
    ('Fixed', FixedVGG),
    ('Piecewise', PW_VGG),
]
seeds = [0, 1, 2, 3]
all_results = {}

for name, model_cls in methods:
    accs = []
    for seed in seeds:
        acc = train_one(model_cls(), name, seed)
        accs.append(float(acc))
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    all_results[name] = {'seeds': {str(s): float(a) for s, a in zip(seeds, accs)},
                          'mean': float(mean_acc), 'std': float(std_acc),
                          'n_seeds': len(seeds)}
    acc_str = ', '.join(['%.4f' % a for a in accs])
    print('\n  %s: %.4f +/- %.4f (seeds: [%s])' % (name, mean_acc, std_acc, acc_str))

# Save
out = {
    'experiment': 'Multi-seed CIFAR-10 VGG-8 reproducibility',
    'config': {'epochs': E, 'batch': B, 'lr': LR, 'optimizer': 'AdamW', 'weight_decay': 5e-4},
    'results': all_results,
    'date': '2026-04-28',
}
out_path = os.path.join(os.path.dirname(__file__), 'results_multiseed.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f'\nSaved: {out_path}')
print(f'\n{"="*60}')
print('  SUMMARY (mean +/- std over 4 seeds)')
print(f'{"="*60}')
for name in ['ReLU', 'EML', 'Gated', 'Fixed', 'Piecewise']:
    r = all_results[name]
    print('  %-12s: %.4f +/- %.4f' % (name, r['mean'], r['std']))
