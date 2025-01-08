"""
RMSNorm

RMSNorm 对 LayerNorm 进行了改进，通过简化计算来提高性能
主要适用于深度学习的transformer模型

RMSNorm是对输入特征归一化时，仅仅使用均方根（root mean square rms）
而不像layer norm那样，对每个特征维度都计算均方根。
rms去除了对均值的依赖 仅通过均方根对输入进行缩放 简化计算

"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        # 使用RMSNorm代替LayerNorm
        self.rms_norm1 = RMSNorm(dim)
        self.rms_norm2 = RMSNorm(dim)
        # self.rms_norm1 = nn.RMSNorm(dim)
        # self.rms_norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        # 残差链接
        x = x + attn_output
        # RMSNorm
        x = self.rms_norm1(x)

        # 前馈神经网络
        mlp_output = self.mlp(x)
        x = x + mlp_output
        # RMSNorm
        x = self.rms_norm2(x)

        return x

if __name__ == '__main__':
    model = TransformerBlock(dim=512,num_heads=8)
    x = torch.randn(1, 10, 512)
    out = model(x)
    print(out.shape)
