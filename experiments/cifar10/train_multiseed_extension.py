"""
Multi-seed extension: GELU, Swish, PReLU on CIFAR-10 VGG-8.
Uses EXACT same training config as train_multiseed.py.
3 additional seeds per method.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time, numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda'
B, E, LR = 128, 30, 0.001

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
    return DataLoader(train_ds, B, shuffle=True, num_workers=0), DataLoader(test_ds, B, shuffle=False, num_workers=0)

# ---- Model builders (same VGG-8 backbone as train_multiseed.py) ----

def conv_block_relu(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
def conv_block_gelu(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.GELU())
def conv_block_swish(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.SiLU())
def conv_block_prelu(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.PReLU())

def make_vgg(conv_fn):
    class VGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                conv_fn(3,64), conv_fn(64,64), nn.MaxPool2d(2),
                conv_fn(64,128), conv_fn(128,128), nn.MaxPool2d(2),
                conv_fn(128,256), conv_fn(256,256), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(256,10)
        def forward(self, x): return self.head(self.body(x).flatten(1))
    return VGG

# ---- Training (EXACT same as train_multiseed.py) ----

def train_one(model, name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = get_loaders()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best = 0; t0 = time.time()
    print('  [%s] seed=%d | %d params' % (name, seed, n))
    for ep in range(1, E+1):
        model.train(); tl, tc, tn = 0., 0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            loss = F.cross_entropy(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += loss.item()*x.size(0); tc += (model(x).argmax(1)==y).sum().item(); tn += x.size(0)
        model.eval(); vl, vc, vn = 0., 0, 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(device), y.to(device); o = model(x)
                vl += F.cross_entropy(o, y).item()*x.size(0); vc += (o.argmax(1)==y).sum().item(); vn += x.size(0)
        sched.step(); best = max(best, vc/vn)
        if ep in [1,10,20,30]:
            print('    ep%d: acc=%.4f best=%.4f' % (ep, vc/vn, best))
    return best

# ---- Run ----
methods = [
    ('GELU', make_vgg(conv_block_gelu)),
    ('Swish', make_vgg(conv_block_swish)),
    ('PReLU', make_vgg(conv_block_prelu)),
]
seeds = [1, 2, 3]
all_results = {}

print('Multi-seed extension: CIFAR-10 VGG-8 (GELU, Swish, PReLU)')
print('Seeds: %s | Epochs: %d | LR: %s' % (str(seeds), E, LR))

for name, model_cls in methods:
    accs = []
    for seed in seeds:
        acc = train_one(model_cls(), name, seed)
        accs.append(float(acc))
    mean_acc = np.mean(accs); std_acc = np.std(accs)
    all_results[name] = {'seeds': {str(s): float(a) for s, a in zip(seeds, accs)},
                          'mean': float(mean_acc), 'std': float(std_acc), 'n_seeds': len(seeds)}
    print('\n  %s: %.4f +/- %.4f (seeds: [%s])' % (name, mean_acc, std_acc, ', '.join(['%.4f'%a for a in accs])))

out_path = os.path.join(os.path.dirname(__file__), 'results_multiseed_extension.json')
with open(out_path, 'w') as f:
    json.dump({'experiment': 'CIFAR-10 VGG-8 multi-seed extension',
               'methods': ['GELU','Swish','PReLU'], 'epochs': E, 'seeds_per_method': len(seeds),
               'results': all_results, 'date': '2026-04-30'}, f, indent=2)

print('\nSaved: %s' % out_path)
for name in ['GELU','Swish','PReLU']:
    r = all_results[name]
    print('  %-6s: %.4f +/- %.4f' % (name, r['mean'], r['std']))
