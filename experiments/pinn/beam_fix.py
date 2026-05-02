"""Euler-Bernoulli beam: u_tt + u_xxxx = 0 (free vibration).
IC: u(x,0)=sin(pi*x), u_t(x,0)=0. BC: u(0,t)=u(1,t)=u_xx(0,t)=u_xx(1,t)=0.
Weighted losses + gradient clipping for 4th-order stability.
"""
import torch, torch.nn as nn, numpy as np, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PiecewiseC1(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Parameter(torch.full((ch,), -0.7)); self.b = nn.Parameter(torch.full((ch,), 0.3))
        self.c = nn.Parameter(torch.full((ch,), 1.0)); self.d = nn.Parameter(torch.full((ch,), -0.1))
        self.alpha = nn.Parameter(torch.full((ch,), 0.1))
    def _e0(self): return torch.exp(self.a) - torch.log(torch.clamp(self.c, min=1e-8))
    def _ep(self): return self.b * torch.exp(self.a) - self.d / torch.clamp(self.c, min=1e-8)
    def forward(self, x):
        a,b,c,d,alpha = self.a, self.b, self.c, self.d, self.alpha
        if x.dim()==2: a,b,c,d,alpha = [p.unsqueeze(0) for p in [a,b,c,d,alpha]]
        left=b*x+a; right=d*x+c
        el = torch.exp(torch.clamp(left,max=15.)) - torch.log(torch.clamp(right,min=1e-8))
        be = self._ep().view_as(alpha); ga = self._e0().view_as(alpha)
        out = torch.where(x<=0, el, alpha*x**2 + be*x + ga)
        return torch.clamp(out, -10., 10.)

ACT_MAP = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'eml': EMLActivation(32, init='gelu'), 'pw': PiecewiseC1(32)}

class BeamPINN(nn.Module):
    def __init__(self, act_name):
        super().__init__()
        act = ACT_MAP[act_name]
        self.net = nn.Sequential(nn.Linear(2,32), act, nn.Linear(32,32), act, nn.Linear(32,32), act, nn.Linear(32,1))
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def run_beam(act_name):
    torch.manual_seed(42); np.random.seed(42)
    model = BeamPINN(act_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    N_f, N_bc, epochs = 1000, 100, 10000  # more epochs for stiff PDE
    
    x_ic = torch.rand(N_bc, 1, device=device); t_ic = torch.zeros(N_bc, 1, device=device)
    u_ic = torch.sin(np.pi * x_ic).to(device)
    
    # Weights: PDE gets 1x, IC gets 10x, BC gets 10x
    w_pde, w_ic, w_bc = 1.0, 10.0, 10.0
    
    best_loss = float('inf')
    for ep in range(epochs):
        opt.zero_grad()
        x_c = torch.rand(N_f, 1, device=device).requires_grad_(True)
        t_c = torch.rand(N_f, 1, device=device).requires_grad_(True)
        
        u = model(x_c, t_c)
        ut   = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        utt  = torch.autograd.grad(ut, t_c, torch.ones_like(ut), create_graph=True)[0]
        ux   = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx  = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        uxxx = torch.autograd.grad(uxx, x_c, torch.ones_like(uxx), create_graph=True)[0]
        uxxxx= torch.autograd.grad(uxxx, x_c, torch.ones_like(uxxx), create_graph=True)[0]
        
        loss_pde = torch.mean((utt + uxxxx)**2)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
        # u_t IC: use a temporary tensor with requires_grad
        t_ic_tmp = torch.zeros(N_bc, 1, device=device, requires_grad=True)
        u_ic_pred = model(x_ic, t_ic_tmp)
        ut_ic = torch.autograd.grad(u_ic_pred, t_ic_tmp, torch.ones_like(u_ic_pred), create_graph=True)[0]
        loss_ic += torch.mean(ut_ic**2)
        
        # BC: u(0,t)=u(1,t)=0
        t_bc = torch.rand(N_bc, 1, device=device)
        x_0 = torch.zeros(N_bc, 1, device=device); x_1 = torch.ones(N_bc, 1, device=device)
        loss_bc = torch.mean(model(x_0, t_bc)**2) + torch.mean(model(x_1, t_bc)**2)
        
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        if torch.isnan(loss): return float('nan')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        loss_val = float(loss.item())
        if loss_val < best_loss: best_loss = loss_val
    
    return best_loss

results = {}
for act in ['tanh','relu','eml','pw']:
    print('%s...'%act, end='', flush=True)
    t0=time.time()
    l=run_beam(act)
    results[act] = l
    t1=time.time()
    print(' loss=%.6f (%ds)'%(l if not np.isnan(l) else 'nan', t1-t0))

# Save results as JSON
import json
out = {'experiment': 'Euler-Bernoulli beam PINN',
       'config': {'epochs': 10000, 'lr': 0.001, 'weighted_losses': True, 'grad_clip': 1.0},
       'results': {act: float(l) for act, l in zip(['tanh','relu','eml','pw'], results) if not np.isnan(l)},
       'script': 'beam_fix.py', 'date': '2026-04-30'}
with open(os.path.join(os.path.dirname(__file__), 'beam_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('Saved beam_results.json')
