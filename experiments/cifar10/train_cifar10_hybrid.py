"""
Hybrid EML Activation: linear asymptotic segment repairs deep degradation
=====================================================================
The root cause of EML failure at depth is exp(bx+a) having no linear asymptote.
Fix: when x > threshold, clamp exp to linear extrapolation, maintaining a hybrid of nonlinear + linear segments.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda'
BATCH, EPOCHS, LR = 128, 30, 0.001

tform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])
tform_t = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])
train_loader = DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'), train=True, download=False, transform=tform), BATCH, shuffle=True, num_workers=0)
test_loader  = DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'), train=False, download=False, transform=tform_t), BATCH, shuffle=False, num_workers=0)


class HybridEMLActivation(nn.Module):
    """
    Hybrid EML: exp(bx+a) switches to linear extrapolation when |bx+a| > threshold
    f(x) = exp_clamped(bx+a) - log(dx+c)
    where exp_clamped(z) = exp(z) for z <= T; exp(T)*(1+z-T) for z > T
    """
    def __init__(self, num_features, threshold=2.0):
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.a = nn.Parameter(torch.full((num_features,), -0.7))
        self.b = nn.Parameter(torch.full((num_features,), 0.4))
        self.c = nn.Parameter(torch.full((num_features,), 1.0))
        self.d = nn.Parameter(torch.full((num_features,), 0.2))

    def forward(self, x):
        if x.dim() == 4:
            a, b, c, d = [p.view(1,-1,1,1) for p in [self.a, self.b, self.c, self.d]]
        else:
            a, b, c, d = [p.view(1,-1) for p in [self.a, self.b, self.c, self.d]]
        z = b * x + a
        T = self.threshold
        # Linear-clamped exp: smooth, keeps gradients
        exp_z = torch.exp(torch.clamp(z, max=T))
        linear_z = torch.exp(torch.tensor(T, device=z.device)) * (z - T + 1.0)
        exp_clamped = torch.where(z <= T, exp_z, linear_z)
        log_term = torch.log(torch.clamp(d * x + c, min=1e-8))
        out = exp_clamped - log_term
        return torch.clamp(out, -10.0, 10.0)


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        HybridEMLActivation(out_c, threshold=2.0),
    )


class HybridVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            conv_block(3, 64), conv_block(64, 64), nn.MaxPool2d(2),
            conv_block(64, 128), conv_block(128, 128), nn.MaxPool2d(2),
            conv_block(128, 256), conv_block(256, 256), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, 10)
    def forward(self, x): return self.head(self.body(x).flatten(1))


model = HybridVGG().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

print(f"\nHybrid EML VGG-8 (linear-clamped exp, T=2.0)")
print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
print(f"{'Ep':>4} {'TrLoss':>10} {'TrAcc':>8} {'TeLoss':>10} {'TeAcc':>8} {'Time':>7}")

best, t0 = 0, time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); tl, tc, tn = 0., 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device); opt.zero_grad()
        loss = F.cross_entropy(model(x), y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tl += loss.item()*x.size(0); tc += (model(x).argmax(1)==y).sum().item(); tn += x.size(0)
    model.eval(); vl, vc, vn = 0., 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device); o = model(x)
            vl += F.cross_entropy(o,y).item()*x.size(0); vc += (o.argmax(1)==y).sum().item(); vn += x.size(0)
    sched.step(); best = max(best, vc/vn)
    print(f"{ep:>4} {tl/tn:>10.4f} {tc/tn:>8.4f} {vl/vn:>10.4f} {vc/vn:>8.4f} {time.time()-t0:>6.0f}s {'*' if vc/vn>=best else ''}")

print(f"\nHybrid EML final: {best:.4f}")
print(f"  vs pure EML:  0.5443  (Δ = {best-0.5443:+.4f})")
print(f"  vs ReLU:      0.9068  (Δ = {best-0.9068:+.4f})")

torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'hybrid_eml.pt'))
print("Saved hybrid_eml.pt")
