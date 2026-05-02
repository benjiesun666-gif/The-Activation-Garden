"""C1 continuity check: unconstrained PiecewiseEML at x=0 after training."""
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import os,sys;sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'));from eml_dl import EMLActivation

device='cuda';B,E,LR=128,20,0.001
tform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tform_t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tr=DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=True,download=False,transform=tform),B,shuffle=True,num_workers=0)
te=DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=False,download=False,transform=tform_t),B,shuffle=False,num_workers=0)

class PWfree(nn.Module):
    def __init__(self,ch):
        super().__init__();self.a=nn.Parameter(torch.full((ch,),-0.7));self.b=nn.Parameter(torch.full((ch,),0.3))
        self.c=nn.Parameter(torch.full((ch,),1.0));self.d=nn.Parameter(torch.full((ch,),-0.1))
        self.alpha=nn.Parameter(torch.full((ch,),0.1));self.beta=nn.Parameter(torch.full((ch,),0.5));self.gamma=nn.Parameter(torch.full((ch,),0.5))
    def forward(self,x):
        a,b,c,d,alpha,beta,gamma=self.a,self.b,self.c,self.d,self.alpha,self.beta,self.gamma
        if x.dim()==4:a,b,c,d,alpha,beta,gamma=[p.view(1,-1,1,1) for p in [a,b,c,d,alpha,beta,gamma]]
        left=b*x+a;right=d*x+c
        eml_out=torch.exp(torch.clamp(left,max=15.))-torch.log(torch.clamp(right,min=1e-8))
        out=torch.where(x<=0,eml_out,alpha*x**2+beta*x+gamma)
        return torch.clamp(out,-10.,10.)

def blk(i,o):return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),PWfree(o))
class VGG(nn.Module):
    def __init__(self):super().__init__();self.b=nn.Sequential(blk(3,64),blk(64,64),nn.MaxPool2d(2),blk(64,128),blk(128,128),nn.MaxPool2d(2),blk(128,256),blk(256,256),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d(1));self.h=nn.Linear(256,10)
    def forward(self,x):return self.h(self.b(x).flatten(1))

m=VGG().to(device);opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,E)
print('Training unconstrained piecewise (20 ep)...')
for ep in range(1,E+1):
    m.train();tl,tc,tn=0.,0,0
    for x,y in tr:x,y=x.to(device),y.to(device);opt.zero_grad();loss=F.cross_entropy(m(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.0);opt.step();tl+=loss.item()*x.size(0);tc+=(m(x).argmax(1)==y).sum().item();tn+=x.size(0)
    m.eval();vl,vc,vn=0.,0,0
    with torch.no_grad():
        for x,y in te:x,y=x.to(device),y.to(device);o=m(x);vl+=F.cross_entropy(o,y).item()*x.size(0);vc+=(o.argmax(1)==y).sum().item();vn+=x.size(0)
    sched.step()
    if ep%5==0:print('  ep%d: acc=%.4f'%(ep,vc/vn))

m.eval();vm_all=[];dm_all=[]
for mod in m.modules():
    if isinstance(mod,PWfree):
        a=mod.a.detach().cpu().numpy();b=mod.b.detach().cpu().numpy()
        c=mod.c.detach().cpu().numpy();d=mod.d.detach().cpu().numpy()
        beta=mod.beta.detach().cpu().numpy();gamma=mod.gamma.detach().cpu().numpy()
        eml0=np.exp(a)-np.log(np.maximum(c,1e-8))
        eml0p=b*np.exp(a)-d/np.maximum(c,1e-8)
        vm_all.extend(np.abs(eml0-gamma).tolist());dm_all.extend(np.abs(eml0p-beta).tolist())

vm=np.array(vm_all);dm=np.array(dm_all)
print('\nC1 continuity at x=0 (%d channels total):' % len(vm))
print('  |EML(0)-gamma|: mean=%.6f std=%.6f max=%.6f'%(vm.mean(),vm.std(),vm.max()))
print('  |EML\'(0)-beta|: mean=%.6f std=%.6f max=%.6f'%(dm.mean(),dm.std(),dm.max()))
print('  Value <0.01: %.1f%%'%(100*(vm<0.01).sum()/len(vm)))
print('  Deriv <0.01: %.1f%%'%(100*(dm<0.01).sum()/len(dm)))
print('  Deriv <0.001: %.1f%%'%(100*(dm<0.001).sum()/len(dm)))

# Save results as JSON
import json, os
out = {'experiment': 'C1 continuity analysis - unconstrained PiecewiseEML',
       'derivative_mismatch_at_x0': {'mean': float(dm.mean()), 'std': float(dm.std()), 'max': float(dm.max()),
                                     'channels_below_0.01_pct': float(100*(dm<0.01).sum()/len(dm)),
                                     'channels_below_0.001_pct': float(100*(dm<0.001).sum()/len(dm))},
       'value_mismatch_at_x0': {'mean': float(vm.mean()), 'std': float(vm.std()), 'max': float(vm.max()),
                                'channels_below_0.01_pct': float(100*(vm<0.01).sum()/len(vm))},
       'total_channels': len(vm),
       'script': 'c1_continuity.py', 'date': '2026-04-30'}
with open(os.path.join(os.path.dirname(__file__), 'cifar10', 'results_c1_continuity.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('Saved: results_c1_continuity.json')
