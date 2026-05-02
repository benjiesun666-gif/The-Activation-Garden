"""
MNIST Benchmark: EML vs ReLU activation functions
=================================================
Tests EMLActivation as a learnable activation function on a standard dataset.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from eml_dl import EMLCNN, ReLUCNN, train_one_epoch, evaluate, count_parameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE, EPOCHS, LR = 128, 10, 0.001

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'), train=True, download=True, transform=tform)
test_ds = datasets.MNIST(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'), train=False, download=True, transform=tform)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

def run(name, model):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n_params = count_parameters(model)
    print(f"\n{'='*60}\n  {name} | {n_params:,} params\n{'='*60}")
    print(f"{'Ep':>4} {'TrLoss':>10} {'TrAcc':>8} {'TeLoss':>10} {'TeAcc':>8} {'Time':>7}")
    best, t0 = 0, time.time()
    for ep in range(1, EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, device)
        vl, va = evaluate(model, test_loader, device)
        sched.step()
        t1 = time.time()
        star = "*" if va > best else ""
        best = max(best, va)
        print(f"{ep:>4} {tl:>10.4f} {ta:>8.4f} {vl:>10.4f} {va:>8.4f} {t1-t0:>6.1f}s{star}")
        t0 = t1
    return best, model

print(f"\nDevice: {device} | Batch: {BATCH_SIZE} | LR: {LR}")
acc_r, _ = run("ReLU CNN", ReLUCNN())
acc_e, eml_model = run("EML CNN", EMLCNN(eml_init='identity'))

print(f"\n{'='*60}")
print(f"  ReLU CNN:  {acc_r:.4f}")
print(f"  EML CNN:   {acc_e:.4f}  (d={acc_e-acc_r:+.4f})")
print(f"{'='*60}")

torch.save(eml_model.state_dict(), 'eml_mnist.pt')
print("Saved eml_mnist.pt")
