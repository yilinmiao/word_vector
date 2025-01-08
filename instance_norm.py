"""
Batch Norm
对每个batch，在特征维度上进行归一化

Layer Norm
对每个样本，在特征维度上进行归一化，不依赖于批次大小

Instance Norm
对每个样本的每个通道在空间维度（H，W）维度上进行归一化，最初用于图像风格迁移任务

Group Norm
将通道分为若干组，对每个样本每个通道，在空间维度上进行归一化
"""

#Batch Norm
import torch
from torch import nn

class CNNWithInstanceNorm(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(784,512)
        # self.ln1 = nn.InstanceNorm2d(512)
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.in1 = nn.InstanceNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        #假设输入的图像大小为32*32
        self.fc = nn.Linear(16*16*16,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = x.reshape(-1,16*16*16)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = CNNWithInstanceNorm()
    x = torch.randn(4,3,32,32)
    out = model(x)
    print(out.shape)
    print(out)