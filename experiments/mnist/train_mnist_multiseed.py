"""
Multi-seed MNIST: 5 seeds to establish statistical patterns in activation differentiation.
For each seed: train EML MNIST, classify activations per layer.
Report mean ± std of exp-like / linear / other proportions.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, numpy as np, time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','..')); from eml_dl import EMLActivation

device='cuda'; B,E,LR=128,10,0.001
tform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

def train_and_classify(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=DataLoader(datasets.MNIST(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=True,download=False,transform=tform),B,shuffle=True,num_workers=0)
    te=DataLoader(datasets.MNIST(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=False,download=False,transform=tform),B,shuffle=False,num_workers=0)
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f=nn.Sequential(
                nn.Conv2d(1,16,3,stride=2,padding=1),nn.BatchNorm2d(16),EMLActivation(16,init='identity'),
                nn.Conv2d(16,32,3,stride=2,padding=1),nn.BatchNorm2d(32),EMLActivation(32,init='identity'),
                nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),EMLActivation(64,init='identity'),
                nn.AdaptiveAvgPool2d(1),
            )
            self.h=nn.Linear(64,10)
        def forward(self,x): return self.h(self.f(x).flatten(1))
    
    m=Net().to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,E)
    for ep in range(1,E+1):
        m.train(); tl,tc,tn=0.,0,0
        for x,y in tr: x,y=x.to(device),y.to(device); opt.zero_grad(); loss=F.cross_entropy(m(x),y); loss.backward(); opt.step(); tl+=loss.item()*x.size(0); tc+=(m(x).argmax(1)==y).sum().item(); tn+=x.size(0)
        m.eval(); vl,vc,vn=0.,0,0
        with torch.no_grad():
            for x,y in te: x,y=x.to(device),y.to(device); o=m(x); vl+=F.cross_entropy(o,y).item()*x.size(0); vc+=(o.argmax(1)==y).sum().item(); vn+=x.size(0)
        sched.step()
    acc=vc/vn
    
    # Classify
    acts=[mod for mod in m.modules() if isinstance(mod,EMLActivation)]
    layer_results={}
    for li,act in enumerate(acts):
        layer_name=['conv1','conv2','conv3'][li]
        a=act.a.detach().cpu().numpy(); b=act.b.detach().cpu().numpy()
        c=act.c.detach().cpu().numpy(); d=act.d.detach().cpu().numpy()
        xv=np.linspace(-4,4,200)
        cats={'exp-like':0,'linear':0,'other':0}
        for i in range(len(a)):
            y=np.exp(np.clip(b[i]*xv+a[i],-80,15))-np.log(np.clip(d[i]*xv+c[i],1e-10,1e20)); y=np.clip(y,-10,10)
            s=np.mean(np.abs(np.gradient(y,xv))); y4=y[-1]
            coeffs=np.polyfit(xv,y,1); yp=np.polyval(coeffs,xv); r2=1-np.sum((y-yp)**2)/(np.sum((y-np.mean(y))**2)+1e-10)
            if s>0.5 and y4>1.0: cats['exp-like']+=1
            elif 0.1<s<0.5 and r2>0.9: cats['linear']+=1
            else: cats['other']+=1
        layer_results[layer_name]=cats
    return seed, acc, layer_results

print(f'Multi-seed MNIST (5 seeds)...')
results=[]
for seed in [0,1,2,3,4]:
    s,acc,layers=train_and_classify(seed)
    results.append({'seed':s,'acc':float(acc),'layers':layers})
    print(f'  Seed {s}: acc={acc:.4f} | c1={layers["conv1"]} c2={layers["conv2"]} c3={layers["conv3"]}')

# Statistics
print(f'\n{"="*60}')
print(f'  STATISTICAL SUMMARY (mean ± std over 5 seeds)')
print(f'{"="*60}')
accs=[r['acc'] for r in results]
print(f'  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
for li,layer in enumerate(['conv1','conv2','conv3']):
    for cat in ['exp-like','linear','other']:
        vals=[r['layers'][layer][cat] for r in results]
        n=16 if li==0 else (32 if li==1 else 64)
        pct_vals=[v/n*100 for v in vals]
        print(f'  {layer} {cat}: {np.mean(vals):.1f}/{n} = {np.mean(pct_vals):.1f}% ± {np.std(pct_vals):.1f}%')

with open(os.path.join(os.path.dirname(__file__), 'multi_seed.json'),'w') as f:
    json.dump({'experiment':'Multi-seed MNIST activation shapes','seeds':5,'results':results,
               'summary':{'mean_acc':float(np.mean(accs)),'std_acc':float(np.std(accs))}},f,indent=2)
print(f'\nSaved: multi_seed.json')
