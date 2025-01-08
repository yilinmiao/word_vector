# CNN +　multihead self attention
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNWithMHS(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Mish(),  # better than ReLU, swish activation function.
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (batch_size, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (batch_size, 64, 7, 7)
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        #classify
        self.fc = nn.Linear(embed_dim, num_classes=10)

    def forward(self, x):
        # cnn
        batch_size = x.size(0)
        cnn_features = self.cnn(x) # (N,64,H/4,W/4)

        # 将特征图展平为序列形式 NCHW==>NSV
        cnn_features = cnn_features.reshape(batch_size,64,-1) # (batch_size, 64, H*W/16)

        # (batch_size,H*W/16,64)
        cnn_features = cnn_features.transpose(1,2)

        # multihead self attention
        attention_output,_ = self.attention(cnn_features) # (batch_size,H*W/16,64)

        # 全局平均池化
        global_avg_pool = attention_output.mean(attention_output,dim=1) # (batch_size,64)  # (batch_size, embed_dim)
        logits = self.fc(global_avg_pool) # (batch_size,10)
        return logits

if __name__ == '__main__':
    x = torch.rand(2, 1, 28, 28)
    model = CNNWithMHS(num_classes=10)
    output = model(x)
    print(output.shape)