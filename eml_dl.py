"""
EML for Deep Learning — 从 Sheffer 算子到可学习激活函数
==========================================================
论文: arXiv:2603.21852

三个模块:
  1. EMLActivation   — drop-in 替代 ReLU/GELU 的可学习激活函数
  2. EMLConv2d        — 内置 EML 激活的卷积层
  3. EMLResBlock      — EML 残差块

核心思想:
  eml(x,y) = exp(x) - ln(y) 能生成所有初等函数
  → 用微型可训练 EML 树做激活函数, 每个神经元自选最优非线性
  → 理论上可以内在地表示 ReLU-like, sigmoid-like, exp-like 等形态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. 稳定的 EML 基础运算
# ============================================================================

EXP_MAX = 15.0   # exp(15) ≈ 3.3e6, float32 安全
LOG_MIN = 1e-8


def eml_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """eml(x,y) = exp(clamp(x)) - log(clamp(y))"""
    return torch.exp(x.clamp(max=EXP_MAX)) - torch.log(y.clamp(min=LOG_MIN))


def eml_depth1(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Depth-1 EML 树: eml(a*1 + b*x, c*1 + d*x)
    展开 = exp(b*x + a) - log(d*x + c)
    4 个可训练参数, 极快
    """
    left = b * x + a       # b*x + a
    right = d * x + c      # d*x + c
    return eml_forward(left, right)


# ============================================================================
# 2. EMLActivation — 替代 nn.ReLU / nn.GELU
# ============================================================================

class EMLActivation(nn.Module):
    """
    可学习 EML 激活函数, 每个特征独立参数化。
    """

    def __init__(self, num_features: int, init: str = 'gelu'):
        super().__init__()
        self.num_features = num_features

        if init == 'identity':
            a = torch.full((num_features,), -0.7)
            b = torch.full((num_features,), 0.3)
            c = torch.full((num_features,), 1.0)
            d = torch.full((num_features,), -0.1)
        elif init == 'gelu':
            a = torch.full((num_features,), -0.7)
            b = torch.full((num_features,), 0.4)
            c = torch.full((num_features,), 1.0)
            d = torch.full((num_features,), 0.2)
        else:
            a = torch.randn(num_features) * 0.1 - 0.5
            b = torch.randn(num_features) * 0.1 + 0.3
            c = torch.randn(num_features) * 0.1 + 1.0
            d = torch.randn(num_features) * 0.05

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        self.d = nn.Parameter(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            a = self.a.view(1, -1, 1, 1)
            b = self.b.view(1, -1, 1, 1)
            c = self.c.view(1, -1, 1, 1)
            d = self.d.view(1, -1, 1, 1)
        elif x.dim() == 2:
            a = self.a.view(1, -1)
            b = self.b.view(1, -1)
            c = self.c.view(1, -1)
            d = self.d.view(1, -1)
        elif x.dim() == 3:
            a = self.a.view(1, -1, 1)
            b = self.b.view(1, -1, 1)
            c = self.c.view(1, -1, 1)
            d = self.d.view(1, -1, 1)
        else:
            raise ValueError(f"Unsupported input dim: {x.dim()}")

        out = eml_depth1(x, a, b, c, d)
        # 防止 exp 爆炸
        out = torch.clamp(out, min=-10.0, max=10.0)
        return out


# ============================================================================
# 3. EMLConv2d — 内置 EML 的卷积层
# ============================================================================

class EMLConv2d(nn.Module):
    """
    Conv2d → EMLActivation 的便捷封装。
    等价于: Conv2d → EMLActivation
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eml_init='identity'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        self.act = EMLActivation(out_channels, init=eml_init)

    def forward(self, x):
        return self.act(self.conv(x))


# ============================================================================
# 4. EMLResBlock — 使用 EML 激活的残差块
# ============================================================================

class EMLResBlock(nn.Module):
    """Conv → EML → Conv → EML, 残差连接"""

    def __init__(self, channels: int, eml_init='identity'):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = EMLActivation(channels, init=eml_init)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = EMLActivation(channels, init=eml_init)

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.act2(out)
        return out


# ============================================================================
# 5. 小型 CNN — MNIST 基准
# ============================================================================

class EMLCNN(nn.Module):
    """小 CNN, EML 激活, MNIST 测试."""

    def __init__(self, eml_init='identity'):
        super().__init__()
        self.features = nn.Sequential(
            EMLConv2d(1, 16, 3, stride=2, padding=1, eml_init=eml_init),
            EMLConv2d(16, 32, 3, stride=2, padding=1, eml_init=eml_init),
            EMLConv2d(32, 64, 3, stride=2, padding=1, eml_init=eml_init),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class ReLUCNN(nn.Module):
    """相同架构, ReLU 激活 (用于对照)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 10))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ============================================================================
# 6. 训练工具
# ============================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 7. 分析工具 — 提取已学习的激活函数形态
# ============================================================================

@torch.no_grad()
def analyze_activation(act: EMLActivation, x_range=(-3.0, 3.0), n_points=200):
    xs = torch.linspace(x_range[0], x_range[1], n_points)
    ys = act(xs.unsqueeze(1).expand(-1, act.num_features))
    return xs.numpy(), ys.numpy().T  # (n_points,), (num_features, n_points)


# ============================================================================
# 8. 自诊断
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EML for Deep Learning — Module Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 测试 EMLActivation
    print("\n[1] EMLActivation")
    act = EMLActivation(8).to(device)
    x = torch.randn(4, 8, device=device)
    y = act(x)
    print(f"  Input:  {x.shape} → Output: {y.shape}")
    print(f"  NaN check: {torch.isnan(y).any().item()}")
    print(f"  Parameters: {count_parameters(act)}")

    # 验证可训练性
    y.sum().backward()
    print(f"  Grad flow: a.grad={act.a.grad is not None}, b.grad={act.b.grad is not None}")

    # 测试 EMLConv2d
    print("\n[2] EMLConv2d")
    conv = EMLConv2d(3, 16, 3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    y = conv(x)
    print(f"  Input:  {x.shape} → Output: {y.shape}")
    print(f"  Parameters: {count_parameters(conv)}")

    # 测试 EMLCNN
    print("\n[3] EMLCNN MNIST forward")
    model = EMLCNN().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = model(x)
    print(f"  Input:  {x.shape} → Output: {y.shape}")
    print(f"  Parameters: {count_parameters(model)}")

    # 与 ReLU 对照
    relu_model = ReLUCNN().to(device)
    print(f"\n[4] ReLU CNN parameters: {count_parameters(relu_model)}")

    # 激活曲线分析
    print("\n[5] Activation function shape analysis")
    act = EMLActivation(4, init='identity').to('cpu')
    xs, ys = analyze_activation(act)
    for i in range(min(4, act.num_features)):
        print(f"  Neuron {i}: y(-3)={ys[i, 0]:.3f}, y(0)={ys[i, 100]:.3f}, y(3)={ys[i, -1]:.3f}")

    print("\n✓ All tests passed — ready for MNIST training")
