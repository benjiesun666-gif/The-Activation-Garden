"""
Scaled EML: exp(α·left) - β·log(right)
α = exponential scaling (0=flat, large=steep)
β = logarithmic scaling (0=pure exponential, large=log correction)
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device='cuda'; B,E,LR=128,30,0.001
tform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tform_t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
tr=DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=True,download=False,transform=tform),B,shuffle=True,num_workers=0)
te=DataLoader(datasets.CIFAR10(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets'),train=False,download=False,transform=tform_t),B,shuffle=False,num_workers=0)

class ScaledEML(nn.Module):
    """f(x) = exp(alpha * (b*x + a)) - beta * log(d*x + c)
    6 params/channel: a,b,c,d basic shape + alpha exp scaling + beta log scaling"""
    def __init__(self,ch,init='gelu'):
        super().__init__()
        self.ch=ch
        if init=='gelu':
            self.a=nn.Parameter(torch.full((ch,),-0.7))
            self.b=nn.Parameter(torch.full((ch,),0.4))
            self.c=nn.Parameter(torch.full((ch,),1.0))
            self.d=nn.Parameter(torch.full((ch,),0.2))
        else:
            self.a=nn.Parameter(torch.randn(ch)*0.1-0.5)
            self.b=nn.Parameter(torch.randn(ch)*0.1+0.3)
            self.c=nn.Parameter(torch.randn(ch)*0.1+1.0)
            self.d=nn.Parameter(torch.randn(ch)*0.05)
        self.alpha=nn.Parameter(torch.full((ch,),1.0))  # exp scaling: 1=standard, <1=flatten, >1=steeper
        self.beta=nn.Parameter(torch.full((ch,),1.0))   # log scaling: 1=standard, 0=pure exponential

    def forward(self,x):
        if x.dim()==4: v=lambda p:p.view(1,-1,1,1)
        else: v=lambda p:p.view(1,-1)
        a,b,c,d,alpha,beta=[v(p) for p in [self.a,self.b,self.c,self.d,self.alpha,self.beta]]
        left=b*x+a
        right=d*x+c
        # alpha controls exponential growth rate
        exp_part=torch.exp(torch.clamp(alpha*left,max=8.0))
        # beta controls logarithmic contribution
        log_part=beta*torch.log(torch.clamp(right,min=1e-8))
        # Linear asymptote: switch to linear extrapolation when left > 2
        T=2.0
        linear_exp=torch.exp(torch.tensor(alpha[0,0,0,0]*T,device=x.device))*(left-T+1.0)
        exp_safe=torch.where(left<=T,exp_part,linear_exp)
        out=exp_safe-log_part
        return torch.clamp(out,-10.0,10.0)

class ResBlock(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.c1=nn.Conv2d(i,o,3,s,padding=1,bias=False); self.b1=nn.BatchNorm2d(o)
        self.c2=nn.Conv2d(o,o,3,1,padding=1,bias=False); self.b2=nn.BatchNorm2d(o)
        self.act1=ScaledEML(o); self.act2=ScaledEML(o)
        self.skip=nn.Sequential()
        if s!=1 or i!=o: self.skip=nn.Sequential(nn.Conv2d(i,o,1,s,bias=False),nn.BatchNorm2d(o))
    def forward(self,x):
        out=self.act1(self.b1(self.c1(x)))
        out=self.b2(self.c2(out))
        out+=self.skip(x)
        return self.act2(out)

class ScaledResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.s=nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),ScaledEML(64))
        self.l1=nn.Sequential(ResBlock(64,64,1),ResBlock(64,64,1))
        self.l2=nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.l3=nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.pool=nn.AdaptiveAvgPool2d(1); self.h=nn.Linear(256,10)
    def forward(self,x):
        x=self.s(x); x=self.l1(x); x=self.l2(x); x=self.l3(x)
        return self.h(self.pool(x).flatten(1))

class ReLUResBlock(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.c1=nn.Conv2d(i,o,3,s,padding=1,bias=False); self.b1=nn.BatchNorm2d(o)
        self.c2=nn.Conv2d(o,o,3,1,padding=1,bias=False); self.b2=nn.BatchNorm2d(o)
        self.act=nn.ReLU(inplace=True)
        self.skip=nn.Sequential()
        if s!=1 or i!=o: self.skip=nn.Sequential(nn.Conv2d(i,o,1,s,bias=False),nn.BatchNorm2d(o))
    def forward(self,x):
        out=self.act(self.b1(self.c1(x)))
        out=self.b2(self.c2(out))
        out+=self.skip(x)
        return self.act(out)

class ReLUScaledResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.s=nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.l1=nn.Sequential(ReLUResBlock(64,64,1),ReLUResBlock(64,64,1))
        self.l2=nn.Sequential(ReLUResBlock(64,128,2),ReLUResBlock(128,128,1))
        self.l3=nn.Sequential(ReLUResBlock(128,256,2),ReLUResBlock(256,256,1))
        self.pool=nn.AdaptiveAvgPool2d(1); self.h=nn.Linear(256,10)
    def forward(self,x):
        x=self.s(x); x=self.l1(x); x=self.l2(x); x=self.l3(x)
        return self.h(self.pool(x).flatten(1))

def train_model(name,model):
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,E)
    n=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n{"="*60}\n  {name} | {n:,} params\n{"="*60}')
    print(f"{'Ep':>4} {'TrLoss':>10} {'TrAcc':>8} {'TeLoss':>10} {'TeAcc':>8} {'Time':>7}")
    best,t0=0,time.time()
    for ep in range(1,E+1):
        model.train(); tl,tc,tn=0.,0,0
        for x,y in tr:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            loss=F.cross_entropy(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            tl+=loss.item()*x.size(0); tc+=(model(x).argmax(1)==y).sum().item(); tn+=x.size(0)
        model.eval(); vl,vc,vn=0.,0,0
        with torch.no_grad():
            for x,y in te:
                x,y=x.to(device),y.to(device); o=model(x)
                vl+=F.cross_entropy(o,y).item()*x.size(0); vc+=(o.argmax(1)==y).sum().item(); vn+=x.size(0)
        sched.step(); best=max(best,vc/vn)
        print(f"{ep:>4} {tl/tn:>10.4f} {tc/tn:>8.4f} {vl/vn:>10.4f} {vc/vn:>8.4f} {time.time()-t0:>6.0f}s {'*' if vc/vn>=best else ''}")
    return best

# --- Run ---
r1=train_model("Scaled EML ResNet-18",ScaledResNet())
r2=train_model("ReLU ResNet-18",ReLUScaledResNet())

print(f"\n{'='*60}")
print(f"  ReLU ResNet-18:   {r2:.4f}")
print(f"  Scaled EML ResNet: {r1:.4f}  (Δ = {r1-r2:+.4f})")
print(f"{'='*60}")

# Show alpha/beta stats
for m in [m for m in ScaledResNet().modules() if isinstance(m,ScaledEML)]:
    a=torch.clamp(m.alpha.detach(),0,5); b_=torch.clamp(m.beta.detach(),0,5)
    print(f"  α∈[{a.min():.2f},{a.max():.2f}] mean={a.mean():.3f}  β∈[{b_.min():.2f},{b_.max():.2f}] mean={b_.mean():.3f}")
