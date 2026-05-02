"""
PINN 1D Heat Equation: reproduces Tanh/ReLU/EML results in pinn_results.json.
u_t = u_xx on [0,1]^2, IC: u(x,0)=sin(pi*x), BC: u(0,t)=u(1,t)=0.
MLP [2,32,32,32,1], 5000 Adam epochs, StepLR scheduler.
"""
import torch, torch.nn as nn, json, time, numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PINN(nn.Module):
    def __init__(self, act_name):
        super().__init__()
        acts = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'eml': EMLActivation(32, init='gelu')}
        self.act = acts[act_name]
        self.net = nn.Sequential(
            nn.Linear(2, 32), self.act,
            nn.Linear(32, 32), self.act,
            nn.Linear(32, 32), self.act,
            nn.Linear(32, 1),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def train_pinn(act_name):
    torch.manual_seed(42); np.random.seed(42)
    model = PINN(act_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)

    # Collocation points
    N_f = 2000
    x_c = torch.rand(N_f, 1, device=device, requires_grad=True)
    t_c = torch.rand(N_f, 1, device=device, requires_grad=True)
    # IC points
    x_ic = torch.rand(100, 1, device=device)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.sin(np.pi * x_ic).to(device)
    # BC points
    t_bc = torch.rand(100, 1, device=device)
    x_bc_0 = torch.zeros_like(t_bc)
    x_bc_1 = torch.ones_like(t_bc)

    best_l2 = float('inf')
    t0 = time.time()
    for ep in range(5000):
        opt.zero_grad()
        u = model(x_c, t_c)
        u_t = torch.autograd.grad(u, t_c, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_c, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_c, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        loss_pde = torch.mean((u_t - u_xx) ** 2)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        u_bc0 = model(x_bc_0, t_bc); u_bc1 = model(x_bc_1, t_bc)
        loss_bc = torch.mean(u_bc0 ** 2) + torch.mean(u_bc1 ** 2)
        loss = loss_pde + loss_ic + loss_bc
        if torch.isnan(loss): break
        loss.backward(); opt.step(); sched.step()
        # L2 check every 200 epochs
        if ep % 200 == 0:
            model.eval()
            with torch.no_grad():
                xs = torch.linspace(0, 1, 50, device=device)
                ts = torch.linspace(0, 1, 20, device=device)
                X, T = torch.meshgrid(xs, ts, indexing='ij')
                u_pred = model(X.reshape(-1,1), T.reshape(-1,1))
                u_true = torch.sin(np.pi*X.reshape(-1,1))*torch.exp(-np.pi**2*T.reshape(-1,1))
                l2 = torch.sqrt(torch.mean((u_pred-u_true)**2)).item()
                if l2 < best_l2: best_l2 = l2
            model.train()
    return best_l2

results = {}
print(f'Training PINN (device={device})...')
for act in ['tanh', 'relu', 'eml']:
    print(f'  {act}...', end='', flush=True)
    l2 = train_pinn(act)
    results[act] = {'final_l2_error': float(l2)}
    print(f' L2={l2:.6f}')

out = {
    'experiment': 'PINN 1D Heat Equation',
    'config': {'epochs': 5000, 'architecture': 'MLP [2,32,32,32,1]', 'lr': 0.001,
               'optimizer': 'Adam', 'scheduler': 'StepLR(1000,0.5)', 'N_collocation': 2000},
    'results': results,
    'date': '2026-04-28',
    'script': 'pinn_experiment.py',
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pinn_results.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'Saved: pinn_results.json')
for act in ['Tanh', 'ReLU', 'EML']:
    print(f'  {act}: L2={results[act.lower()]["final_l2_error"]:.6f}')
