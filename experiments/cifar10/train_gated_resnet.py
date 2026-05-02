"""
Task 1: Gated EML on ResNet-18 CIFAR-10
=======================================
GatedEML: sigma(gate) * EMLActivation(x) + (1-sigma(gate)) * ReLU(x)
One learnable gate per channel per ResBlock.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, json, os, sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda'; B, E, LR = 128, 30, 0.001

tform = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tform_t = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
train_loader = DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=True,download=False,transform=tform),B,shuffle=True,num_workers=0)
test_loader  = DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=False,download=False,transform=tform_t),B,shuffle=False,num_workers=0)

class GatedEML(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.eml = EMLActivation(ch, init='gelu')
        self.relu = nn.ReLU(inplace=False)
        self.gate = nn.Parameter(torch.full((ch,), 0.5))
    def forward(self, x):
        g = torch.sigmoid(self.gate)
        if x.dim() == 4: g = g.view(1, -1, 1, 1)
        else: g = g.view(1, -1)
        return g * self.eml(x) + (1 - g) * self.relu(x)

class ResBlock(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, padding=1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, padding=1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.act1 = GatedEML(o); self.act2 = GatedEML(o)
        self.skip = nn.Sequential()
        if s != 1 or i != o:
            self.skip = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))
    def forward(self, x):
        out = self.act1(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        out += self.skip(x)
        return self.act2(out)

class GatedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),GatedEML(64))
        self.l1 = nn.Sequential(ResBlock(64,64,1),ResBlock(64,64,1))
        self.l2 = nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.l3 = nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.pool = nn.AdaptiveAvgPool2d(1); self.h = nn.Linear(256,10)
    def forward(self, x):
        x = self.s(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.h(self.pool(x).flatten(1))

model = GatedResNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*60}")
print(f"  Gated EML ResNet-18 | CIFAR-10 | {n_params:,} params")
print(f"{'='*60}")
print(f"{'Ep':>4} {'TrLoss':>10} {'TrAcc':>8} {'TeLoss':>10} {'TeAcc':>8} {'Time':>7}")

history = []
best, t0 = 0, time.time()
for ep in range(1, E + 1):
    model.train(); tl, tc, tn = 0., 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device); opt.zero_grad()
        loss = F.cross_entropy(model(x), y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tl += loss.item() * x.size(0); tc += (model(x).argmax(1) == y).sum().item(); tn += x.size(0)

    model.eval(); vl, vc, vn = 0., 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device); o = model(x)
            vl += F.cross_entropy(o, y).item() * x.size(0); vc += (o.argmax(1) == y).sum().item(); vn += x.size(0)

    sched.step(); best = max(best, vc / vn)
    elapsed = time.time() - t0
    history.append({'epoch': ep, 'train_loss': tl/tn, 'train_acc': tc/tn, 'test_loss': vl/vn, 'test_acc': vc/vn})
    print(f"{ep:>4} {tl/tn:>10.4f} {tc/tn:>8.4f} {vl/vn:>10.4f} {vc/vn:>8.4f} {elapsed:>6.0f}s {'*' if vc/vn >= best else ''}")

# Gate analysis
gates_info = []
for i, m in enumerate(model.modules()):
    if isinstance(m, GatedEML):
        g = torch.sigmoid(m.gate).detach()
        gates_info.append({'layer_idx': i, 'mean': float(g.mean()), 'std': float(g.std()),
                           'min': float(g.min()), 'max': float(g.max())})

result = {
    'experiment': 'Gated EML ResNet-18 CIFAR-10',
    'architecture': 'ResNet-18 with GatedEML (EML+ReLU gated per channel)',
    'best_accuracy': best,
    'vs_relu_resnet': best - 0.9184,
    'parameters': n_params,
    'epochs': E,
    'history': history,
    'gates': gates_info,
    'date': '2026-04-27',
}

with open(os.path.join(os.path.dirname(__file__), 'results_gated_resnet.json'), 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nBest: {best:.4f} | vs ReLU ResNet-18 (0.9184): {best-0.9184:+.4f}")
print(f"\nGate values:")
for g in gates_info:
    print(f"  mean={g['mean']:.4f} std={g['std']:.4f}")

torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'gated_resnet.pt'))
print("Saved: results_gated_resnet.json, gated_resnet.pt")
