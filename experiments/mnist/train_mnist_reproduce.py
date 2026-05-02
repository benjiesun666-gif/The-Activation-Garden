"""
Reproduce Table 1: Train EML MNIST, classify activation shapes, save JSON.
Uses EXACT paper classification from Methods section.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, numpy as np, time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','..')); from eml_dl import EMLActivation

device='cuda'; B,E,LR=128,10,0.001
tform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
tr=DataLoader(datasets.MNIST(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=True,download=False,transform=tform),B,shuffle=True,num_workers=0)
te=DataLoader(datasets.MNIST(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=False,download=False,transform=tform),B,shuffle=False,num_workers=0)

class EML_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat=nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),nn.BatchNorm2d(16),EMLActivation(16,init='identity'),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.BatchNorm2d(32),EMLActivation(32,init='identity'),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),EMLActivation(64,init='identity'),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head=nn.Linear(64,10)
    def forward(self,x): return self.head(self.feat(x).flatten(1))

m=EML_CNN().to(device)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,E)
print(f'Training EML MNIST ({sum(p.numel() for p in m.parameters() if p.requires_grad):,} params)')
t0=time.time()
for ep in range(1,E+1):
    m.train(); tl,tc,tn=0.,0,0
    for x,y in tr:
        x,y=x.to(device),y.to(device); opt.zero_grad()
        loss=F.cross_entropy(m(x),y); loss.backward(); opt.step()
        tl+=loss.item()*x.size(0); tc+=(m(x).argmax(1)==y).sum().item(); tn+=x.size(0)
    m.eval(); vl,vc,vn=0.,0,0
    with torch.no_grad():
        for x,y in te: x,y=x.to(device),y.to(device); o=m(x); vl+=F.cross_entropy(o,y).item()*x.size(0); vc+=(o.argmax(1)==y).sum().item(); vn+=x.size(0)
    sched.step()
    print(f'{ep:3d}: tr={tc/tn:.4f} te={vc/vn:.4f} | {time.time()-t0:.0f}s')
acc=vc/vn
torch.save(m.state_dict(),os.path.join(os.path.dirname(__file__),'eml_mnist_v2.pt'))

# Classify activation shapes using EXACT paper criteria from Methods
print('\nClassifying activation shapes (paper criteria: slope>0.5 & y(4)>1.0 → exp-like)...')
acts=[mod for mod in m.modules() if isinstance(mod,EMLActivation)]
results={}
for li,act in enumerate(acts):
    layer_name=['conv1','conv2','conv3'][li]
    a=act.a.detach().cpu().numpy(); b=act.b.detach().cpu().numpy()
    c=act.c.detach().cpu().numpy(); d=act.d.detach().cpu().numpy()
    xv=np.linspace(-4,4,200)
    cats={'exp-like':0,'linear':0,'other':0}; channels=[]
    for i in range(len(a)):
        y=np.exp(np.clip(b[i]*xv+a[i],-80,15))-np.log(np.clip(d[i]*xv+c[i],1e-10,1e20))
        y=np.clip(y,-10,10)
        slope=np.mean(np.abs(np.gradient(y,xv)))
        y4=y[-1]; coeffs=np.polyfit(xv,y,1); yp=np.polyval(coeffs,xv)
        r2=1-np.sum((y-yp)**2)/(np.sum((y-np.mean(y))**2)+1e-10)
        if slope>0.5 and y4>1.0: cat='exp-like'
        elif 0.1<slope<0.5 and r2>0.9: cat='linear'
        else: cat='other'
        cats[cat]+=1
        channels.append({'ch':i,'cat':cat,'a':float(a[i]),'b':float(b[i]),'c':float(c[i]),'d':float(d[i]),
                         'slope':float(slope),'y(-4)':float(y[0]),'y(0)':float(y[100]),'y(4)':float(y[-1])})
    results[layer_name]={'counts':cats,'channels':channels,'a_range':[float(a.min()),float(a.max())],'b_range':[float(b.min()),float(b.max())]}
    print(f'  {layer_name}: {cats}')

output={'experiment':'MNIST EML activation shape classification (reproduced)',
        'test_accuracy':float(acc),'classification_criteria':'Slope>0.5 & y(4)>1.0 → exp-like; 0.1<slope<0.5 & R2>0.9 → linear; else other',
        'results':results,'date':'2026-04-27'}
with open(os.path.join(os.path.dirname(__file__),'activation_shapes.json'),'w') as f:
    json.dump(output,f,indent=2)
print(f'Saved: activation_shapes.json, eml_mnist_v2.pt (acc={acc:.4f})')
