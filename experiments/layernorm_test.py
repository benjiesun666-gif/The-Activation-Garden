"""
Bullet #3: LayerNorm vs BatchNorm — does EML still fail under LayerNorm?
Train EML VGG-8 with LayerNorm on CIFAR-10, compare to BatchNorm baseline.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..')); from eml_dl import EMLActivation

device='cuda'; B,E,LR=128,20,0.001
tform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tform_t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tr=DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=True,download=False,transform=tform),B,shuffle=True,num_workers=0)
te=DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=False,download=False,transform=tform_t),B,shuffle=False,num_workers=0)

def train_eml_vgg(norm_type):
    def blk(i,o):
        layers=[nn.Conv2d(i,o,3,padding=1,bias=False)]
        if norm_type=='bn': layers.append(nn.BatchNorm2d(o))
        elif norm_type=='ln': layers.append(nn.InstanceNorm2d(o))  # Per-channel per-sample norm, works on any spatial size
        else: pass  # No normalization
        layers.append(EMLActivation(o,init='identity'))
        return nn.Sequential(*layers)
    class VGG(nn.Module):
        def __init__(self): super().__init__(); self.b=nn.Sequential(blk(3,64),blk(64,64),nn.MaxPool2d(2),blk(64,128),blk(128,128),nn.MaxPool2d(2),blk(128,256),blk(256,256),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d(1)); self.h=nn.Linear(256,10)
        def forward(self,x): return self.h(self.b(x).flatten(1))
    m=VGG().to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=0.001,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,E)
    best=0
    print(f'\n  EML VGG-8 + {norm_type.upper()}')
    for ep in range(1,E+1):
        m.train(); tl,tc,tn=0.,0,0
        for x,y in tr: x,y=x.to(device),y.to(device); opt.zero_grad(); loss=F.cross_entropy(m(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); tl+=loss.item()*x.size(0); tc+=(m(x).argmax(1)==y).sum().item(); tn+=x.size(0)
        m.eval(); vl,vc,vn=0.,0,0
        with torch.no_grad():
            for x,y in te: x,y=x.to(device),y.to(device); o=m(x); vl+=F.cross_entropy(o,y).item()*x.size(0); vc+=(o.argmax(1)==y).sum().item(); vn+=x.size(0)
        sched.step(); best=max(best,vc/vn)
        if ep%5==0: print(f'    Ep{ep:2d}: acc={vc/vn:.4f} (best={best:.4f})')
    return best

acc_bn=train_eml_vgg('bn')
acc_in=train_eml_vgg('ln')
acc_none=train_eml_vgg('none')

result={'experiment':'Normalization comparison for EML VGG-8','epochs':20,
        'BatchNorm':float(acc_bn),'InstanceNorm':float(acc_in),'NoNorm':float(acc_none),
        'date':'2026-04-27'}
with open(os.path.join(os.path.dirname(__file__), 'cifar10', 'results_layernorm.json'),'w') as f:
    json.dump(result,f,indent=2)
print(f'\nBatchNorm: {acc_bn:.4f}, InstanceNorm: {acc_in:.4f}, NoNorm: {acc_none:.4f}')
print('Saved: results_layernorm.json')
