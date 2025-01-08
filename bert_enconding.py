#Bert绝对位置编码
import torch
import torch.nn as nn
import math

# Define position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: 词嵌入维度（每个词向量维度）
        :param max_len: 最大序列长度
        """
        super().__init__()
        # init position encoding matrix.
        # pe stores position encoding.
        pe = torch.zeros(max_len, d_model)
        print(f"初始矩阵pe.shape: {pe.shape}\npe:{pe}")

        # gen position index: shape: (max_length, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        print(f"position.shape: {position.shape}\nposition:{position}")

        # gen div_term: shape: (d_model/2,): 生成从0到d_model-1的偶数序列
        div_term = torch.exp(torch.arange(0, d_model, 2). float() * (-math.log(10000.0) / d_model))
        print(f"频率项：div_term.shape: {div_term.shape}\ndiv_term:{div_term}")

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        print(f"even position shape:{pe[:,0::2].shape}\n pe:{pe[:,0::2]}")
        print(f"odd position shape:{pe[:,1::2].shape}\n pe:{pe[:,1::2]}")

        self.register_buffer('pe', pe.unsqueeze(dim=0))
    def forward(self, x):
        """
        :param x: shape: (batch_size, seq_len, d_model)
        :return: shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:,0:x.size(1)]
        print(self.pe[:,0:x.size(1)].shape)
        # print(x.shape)
        # print(self.pe.shape)
        return x

if __name__ == '__main__':
    d_model = 512 # embedding dimension
    max_len = 100
    pos_encoding = PositionalEncoding(d_model, max_len)

    batch_size = 32
    seq_len = 50
    input_embedding = torch.randn(batch_size, seq_len, d_model)

    out = pos_encoding(input_embedding)
    print(input_embedding.shape)
    print(input_embedding)
    print(out.shape)
    print(out)

