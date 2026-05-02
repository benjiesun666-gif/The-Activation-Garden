"""
PINN with PiecewiseEML activation (C1 constrained at x=0).
Same method as pinn_experiment.py, added PiecewiseEML to act_map.
"""
import torch, torch.nn as nn, json, time, numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PiecewiseC1(nn.Module):
    """EML(x) for x<=0, quadratic C1 for x>0."""
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Parameter(torch.full((ch,), -0.7))
        self.b = nn.Parameter(torch.full((ch,), 0.3))
        self.c = nn.Parameter(torch.full((ch,), 1.0))
        self.d = nn.Parameter(torch.full((ch,), -0.1))
        self.alpha = nn.Parameter(torch.full((ch,), 0.1))
    def _eml0(self):
        return torch.exp(self.a) - torch.log(torch.clamp(self.c, min=1e-8))
    def _eml0p(self):
        return self.b * torch.exp(self.a) - self.d / torch.clamp(self.c, min=1e-8)
    def forward(self, x):
        a, b, c, d, alpha = self.a, self.b, self.c, self.d, self.alpha
        if x.dim() == 2:
            a, b, c, d, alpha = [p.unsqueeze(0) for p in [a, b, c, d, alpha]]
        left = b * x + a; right = d * x + c
        eml_out = torch.exp(torch.clamp(left, max=15.)) - torch.log(torch.clamp(right, min=1e-8))
        beta = self._eml0p().view_as(alpha)
        gamma = self._eml0().view_as(alpha)
        out = torch.where(x <= 0, eml_out, alpha * x**2 + beta * x + gamma)
        return torch.clamp(out, -10., 10.)

class PINN(nn.Module):
    def __init__(self, act_name, layers):
        super().__init__()
        acts = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'eml': EMLActivation(layers[1], init='gelu'),
                'pw': PiecewiseC1(layers[1])}
        self.act = acts[act_name]
        net = []
        for i in range(len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2: net.append(self.act)
        self.net = nn.Sequential(*net)
    def forward(self, *args):
        return self.net(torch.cat(args, dim=1))

def run_heat(act_name, epochs=5000):
    torch.manual_seed(42); np.random.seed(42)
    model = PINN(act_name, [2,32,32,32,1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    N_f, N_bc = 2000, 100
    x_ic = torch.rand(N_bc,1,device=device); t_ic = torch.zeros(N_bc,1,device=device)
    u_ic = torch.sin(np.pi*x_ic).to(device)
    t_bc = torch.rand(N_bc,1,device=device); x_0 = torch.zeros(N_bc,1,device=device); x_1 = torch.ones(N_bc,1,device=device)
    for ep in range(epochs):
        opt.zero_grad()
        x_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        t_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        u = model(x_c, t_c)
        ut = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        loss = torch.mean((ut - uxx)**2) + torch.mean((model(x_ic,t_ic) - u_ic)**2) + torch.mean(model(x_0,t_bc)**2) + torch.mean(model(x_1,t_bc)**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        xs=torch.linspace(0,1,50,device=device); ts=torch.linspace(0,1,20,device=device)
        X,T=torch.meshgrid(xs,ts,indexing='ij')
        up=model(X.reshape(-1,1),T.reshape(-1,1))
        ut=torch.sin(np.pi*X.reshape(-1,1))*torch.exp(-np.pi**2*T.reshape(-1,1))
        l2=torch.sqrt(torch.mean((up-ut)**2)).item()
    return l2

def run_burgers(act_name):
    torch.manual_seed(42); np.random.seed(42)
    model = PINN(act_name, [2,32,32,32,1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    nu = 0.01/np.pi; N_f, N_bc = 2000, 100
    x_ic = torch.rand(N_bc,1,device=device)*2-1; t_ic = torch.zeros(N_bc,1,device=device)
    u_ic = (-torch.sin(np.pi*x_ic)).to(device)
    t_bc = torch.rand(N_bc,1,device=device); x_l = -torch.ones(N_bc,1,device=device); x_r = torch.ones(N_bc,1,device=device)
    for ep in range(5000):
        opt.zero_grad()
        x_c = (torch.rand(N_f,1,device=device)*2-1).requires_grad_(True)
        t_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        u = model(x_c, t_c)
        ut = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        loss = torch.mean((ut + u*ux - nu*uxx)**2) + torch.mean((model(x_ic,t_ic) - u_ic)**2) + torch.mean(model(x_l,t_bc)**2) + torch.mean(model(x_r,t_bc)**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    return float(loss.item())

def run_allencahn(act_name):
    torch.manual_seed(42); np.random.seed(42)
    model = PINN(act_name, [2,32,32,32,1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    eps = 0.001; N_f, N_bc = 2000, 100
    x_ic = torch.rand(N_bc,1,device=device)*2-1; t_ic = torch.zeros(N_bc,1,device=device)
    u_ic = (x_ic**2 * torch.cos(np.pi*x_ic)).to(device)
    t_bc = torch.rand(N_bc,1,device=device); x_l = -torch.ones(N_bc,1,device=device); x_r = torch.ones(N_bc,1,device=device)
    for ep in range(5000):
        opt.zero_grad()
        x_c = (torch.rand(N_f,1,device=device)*2-1).requires_grad_(True)
        t_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        u = model(x_c, t_c)
        ut = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        loss = torch.mean((ut - eps*uxx - u + u**3)**2) + torch.mean((model(x_ic,t_ic) - u_ic)**2) + torch.mean((model(x_l,t_bc) - model(x_r,t_bc))**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    return float(loss.item())

def run_lorenz(act_name):
    torch.manual_seed(42); np.random.seed(42)
    model = PINN(act_name, [1,32,32,32,3]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    s, r, b = 10., 28., 8./3.; N_f, N_bc = 2000, 100
    ic = torch.tensor([1.,1.,1.], device=device).unsqueeze(0)
    for ep in range(5000):
        opt.zero_grad()
        t_c = (torch.rand(N_f,1,device=device)*10.).requires_grad_(True)
        out = model(t_c); xv, yv, zv = out[:,0:1], out[:,1:2], out[:,2:3]
        dx=torch.autograd.grad(xv,t_c,torch.ones_like(xv),create_graph=True)[0]
        dy=torch.autograd.grad(yv,t_c,torch.ones_like(yv),create_graph=True)[0]
        dz=torch.autograd.grad(zv,t_c,torch.ones_like(zv),create_graph=True)[0]
        t0=torch.zeros(N_bc,1,device=device)
        loss=torch.mean((dx - s*(yv - xv))**2)+torch.mean((dy - (xv*(r - zv) - yv))**2)+torch.mean((dz - (xv*yv - b*zv))**2)+torch.mean((model(t0) - ic.expand(N_bc,-1))**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        ts=torch.linspace(0,10,200,device=device).unsqueeze(1)
        pred=model(ts).cpu().numpy()
        ref=np.zeros((200,3)); ref[0]=[1,1,1]; dt=10./199
        for i in range(1,200):
            x,y,z=ref[i-1]
            k=[np.array([s*(y-x), x*(r-z)-y, x*y-b*z])]
            for j in range(3):
                hk=ref[i-1]+0.5*dt*k[-1]
                k.append(np.array([s*(hk[1]-hk[0]),hk[0]*(r-hk[2])-hk[1],hk[0]*hk[1]-b*hk[2]]))
            ref[i]=ref[i-1]+dt/6*(k[0]+2*k[1]+2*k[2]+k[3])
        l2=float(torch.sqrt(torch.mean((torch.tensor(pred)-torch.tensor(ref).float())**2)).item())
    return l2

# ---- Run ----
pdes=[('Heat',run_heat),('Burgers',run_burgers),('AllenCahn',run_allencahn),('Lorenz',run_lorenz)]
results={}
print('PINN + PiecewiseC1 | Device: %s'%device)
for pde_name, fn in pdes:
    print('\n' + '='*50 + '\n  %s\n'%pde_name + '='*50)
    results[pde_name] = {}
    t0=time.time()
    l2=fn('pw')
    t1=time.time()
    results[pde_name]={'l2':round(l2,6) if not np.isnan(l2) else 'nan','time':round(t1-t0,1)}
    s='%.6f'%l2 if not np.isnan(l2) else 'nan'
    print('  PW: L2=%s (%ds)'%(s,t1-t0))

out={'experiment':'PINN + PiecewiseC1','config':{'epochs':5000,'lr':0.001},'results':results,'date':'2026-04-30'}
with open(os.path.join(os.path.dirname(__file__),'pinn_pw_results.json'),'w') as f: json.dump(out,f,indent=2)
print('\nSaved pinn_pw_results.json')
