import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        #q, k, v
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = (torch.matmul(Q,K.transpose(2,1)) / (self.embed_dim**0.5))

        attention_weights = torch.softmax(attention_scores, dim=1)
        out = torch.matmul(attention_weights, V)
        return out,attention_weights

if __name__ == "__main__":
    # n,s,v = batch_size, seq_len, embedding_dim
    x = torch.randn(3,5,10)
    # embedding_dim
    attn = Attention(10)
    out, weights = attn(x)
    # print(out)
    # print(weights)
    print(out.shape)
    print(weights.shape)
    print(torch.sum(weights, dim=1))