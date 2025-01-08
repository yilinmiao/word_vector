import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim,"embed_dim must be divided by num_head"

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)
        self.fc_out = nn.Linear(embed_dim,embed_dim)

    def forward(self, x):
        batch_size,seq_len,embed_dim = x.size()
        Q = self.query(x).reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        K = self.key(x).reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        V = self.value(x).reshape(batch_size,seq_len,self.num_heads,self.head_dim)

        #(batch_size,seq_len,num_heads,head_dim)==>
        #(batch_size,num_heads,seq_len,head_dim)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = torch.matmul(Q,K.transpose(2,3)) / (self.head_dim**0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights,V)

        output = output.transpose(1,2).contiguous().reshape(batch_size,seq_len,embed_dim)
        output = self.fc_out(output)

        return output

if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    embed_dim = 10
    num_heads = 2

    x = torch.rand(batch_size, seq_len, embed_dim)
    attention = MultiHeadSelfAttention(embed_dim, num_heads)
    output = attention(x)
    print(output.shape)

