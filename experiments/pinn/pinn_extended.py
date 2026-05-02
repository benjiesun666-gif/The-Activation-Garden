"""
Extended PINN: Burgers, Allen-Cahn, Euler-Bernoulli, Lorenz.
EXACT SAME method as pinn_experiment.py.
"""
import torch, torch.nn as nn, json, time, numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from eml_dl import EMLActivation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ACT_MAP = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'eml': None}

def make_model(act_name, layers):
    act = ACT_MAP[act_name] if act_name != 'eml' else EMLActivation(layers[1], init='gelu')
    net = []
    for i in range(len(layers)-1):
        net.append(nn.Linear(layers[i], layers[i+1]))
        if i < len(layers)-2: net.append(act)
    return nn.Sequential(*net).to(device)

def run_burgers(act_name, epochs=5000, N_f=2000, N_bc=100):
    torch.manual_seed(42); np.random.seed(42)
    model = make_model(act_name, [2,32,32,32,1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    nu = 0.01 / np.pi
    x_ic = torch.rand(N_bc,1,device=device)*2-1; t_ic = torch.zeros(N_bc,1,device=device)
    u_ic = (-torch.sin(np.pi*x_ic)).to(device)
    t_bc = torch.rand(N_bc,1,device=device); x_l = -torch.ones(N_bc,1,device=device); x_r = torch.ones(N_bc,1,device=device)
    for ep in range(epochs):
        opt.zero_grad()
        x_c = (torch.rand(N_f,1,device=device)*2-1).requires_grad_(True)
        t_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        u = model(torch.cat([x_c,t_c],1))
        ut = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        loss = torch.mean((ut + u * ux - nu * uxx)**2) + torch.mean((model(torch.cat([x_ic,t_ic],1)) - u_ic)**2) + torch.mean(model(torch.cat([x_l,t_bc],1))**2) + torch.mean(model(torch.cat([x_r,t_bc],1))**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    return float(loss.item())

def run_allencahn(act_name, epochs=5000, N_f=2000, N_bc=100):
    torch.manual_seed(42); np.random.seed(42)
    model = make_model(act_name, [2,32,32,32,1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    eps = 0.001
    x_ic = torch.rand(N_bc,1,device=device)*2-1; t_ic = torch.zeros(N_bc,1,device=device)
    u_ic = (x_ic**2 * torch.cos(np.pi*x_ic)).to(device)
    t_bc = torch.rand(N_bc,1,device=device); x_l = -torch.ones(N_bc,1,device=device); x_r = torch.ones(N_bc,1,device=device)
    for ep in range(epochs):
        opt.zero_grad()
        x_c = (torch.rand(N_f,1,device=device)*2-1).requires_grad_(True)
        t_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        u = model(torch.cat([x_c,t_c],1))
        ut = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        loss = torch.mean((ut - eps*uxx - u + u**3)**2) + torch.mean((model(torch.cat([x_ic,t_ic],1)) - u_ic)**2) + torch.mean((model(torch.cat([x_l,t_bc],1)) - model(torch.cat([x_r,t_bc],1)))**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    return float(loss.item())

def run_euler_beam(act_name, epochs=5000, N_f=2000, N_bc=100):
    torch.manual_seed(42); np.random.seed(42)
    model = make_model(act_name, [2,32,32,32,1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    x_ic = torch.rand(N_bc,1,device=device); t_ic = torch.zeros(N_bc,1,device=device)
    u_ic = torch.sin(np.pi*x_ic).to(device)
    t_bc = torch.rand(N_bc,1,device=device); x_0 = torch.zeros(N_bc,1,device=device); x_1 = torch.ones(N_bc,1,device=device)
    for ep in range(epochs):
        opt.zero_grad()
        x_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        t_c = torch.rand(N_f,1,device=device).requires_grad_(True)
        u = model(torch.cat([x_c,t_c],1))
        ut = torch.autograd.grad(u, t_c, torch.ones_like(u), create_graph=True)[0]
        utt = torch.autograd.grad(ut, t_c, torch.ones_like(ut), create_graph=True)[0]
        ux = torch.autograd.grad(u, x_c, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x_c, torch.ones_like(ux), create_graph=True)[0]
        uxxx = torch.autograd.grad(uxx, x_c, torch.ones_like(uxx), create_graph=True)[0]
        uxxxx = torch.autograd.grad(uxxx, x_c, torch.ones_like(uxxx), create_graph=True)[0]
        force = torch.cos(2*np.pi*t_c.detach())*torch.sin(np.pi*x_c.detach()).to(device)
        loss = torch.mean((utt + uxxxx - force)**2) + torch.mean((model(torch.cat([x_ic,t_ic],1)) - u_ic)**2) + torch.mean(model(torch.cat([x_0,t_bc],1))**2) + torch.mean(model(torch.cat([x_1,t_bc],1))**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    return float('nan')

def run_lorenz(act_name, epochs=5000, N_f=2000, N_bc=100):
    torch.manual_seed(42); np.random.seed(42)
    model = make_model(act_name, [1,32,32,32,3])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, 0.5)
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    ic = torch.tensor([1.,1.,1.], device=device).unsqueeze(0)
    for ep in range(epochs):
        opt.zero_grad()
        t_c = (torch.rand(N_f,1,device=device)*10.).requires_grad_(True)
        out = model(t_c)
        xv, yv, zv = out[:,0:1], out[:,1:2], out[:,2:3]
        dx = torch.autograd.grad(xv, t_c, torch.ones_like(xv), create_graph=True)[0]
        dy = torch.autograd.grad(yv, t_c, torch.ones_like(yv), create_graph=True)[0]
        dz = torch.autograd.grad(zv, t_c, torch.ones_like(zv), create_graph=True)[0]
        t0 = torch.zeros(N_bc,1,device=device)
        loss = torch.mean((dx - sigma*(yv - xv))**2) + torch.mean((dy - (xv*(rho - zv) - yv)))**2 + torch.mean((dz - (xv*yv - beta*zv))**2) + torch.mean((model(t0) - ic.expand(N_bc,-1))**2)
        if torch.isnan(loss): return float('nan')
        loss.backward(); opt.step(); sched.step()
    # L2 against RK4
    model.eval()
    with torch.no_grad():
        ts = torch.linspace(0,10,200,device=device).unsqueeze(1)
        pred = model(ts).cpu().numpy()
        ref = np.zeros((200,3)); ref[0] = [1,1,1]; dt = 10./199
        for i in range(1,200):
            x,y,z = ref[i-1]
            k = [np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])]
            for j in range(3):
                hk = ref[i-1] + 0.5*dt*k[-1]
                k.append(np.array([sigma*(hk[1]-hk[0]), hk[0]*(rho-hk[2])-hk[1], hk[0]*hk[1]-beta*hk[2]]))
            ref[i] = ref[i-1] + dt/6*(k[0]+2*k[1]+2*k[2]+k[3])
        l2 = float(torch.sqrt(torch.mean((torch.tensor(pred) - torch.tensor(ref).float())**2)).item())
    return l2

# Run
pdes = ['burgers','allencahn','euler_beam','lorenz']
acts = ['tanh','relu','eml']
results = {}
print('Extended PINN | Device: %s | Epochs: 5000' % device)
for pde in pdes:
    print('\n' + '='*50 + '\n  %s\n'%pde.upper() + '='*50)
    results[pde] = {}
    for act in acts:
        print('  %s...'%act, end='', flush=True)
        t0=time.time()
        if pde=='burgers': l2=run_burgers(act)
        elif pde=='allencahn': l2=run_allencahn(act)
        elif pde=='euler_beam': l2=run_euler_beam(act)
        else: l2=run_lorenz(act)
        t1=time.time()
        results[pde][act]={'l2':round(l2,6) if not np.isnan(l2) else 'nan','time':round(t1-t0,1)}
        s='%.6f'%l2 if not np.isnan(l2) else 'nan'
        print(' L2=%s (%ds)'%(s,t1-t0))

out={'experiment':'Extended PINN','config':{'epochs':5000,'lr':0.001},'results':results,'date':'2026-04-30'}
with open(os.path.join(os.path.dirname(__file__),'pinn_extended_results.json'),'w') as f: json.dump(out,f,indent=2)
print('\nSaved.')
