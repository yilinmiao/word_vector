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

    def forward(self, q_input, k_input, v_input):
        print(f"q_input: {q_input.shape}")
        print(f"v_input: {v_input.shape}")
        print(f"k_input: {k_input.shape}")
        Q = self.query(q_input)
        K = self.key(v_input)
        V = self.value(k_input)
        print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        attention_scores = (torch.matmul(Q,K.transpose(2,1)) / (self.embed_dim**0.5))

        attention_weights = torch.softmax(attention_scores, dim=1)
        out = torch.matmul(attention_weights, V)
        return out,attention_weights

if __name__ == "__main__":
    # n,s,v = batch_size, seq_len, embedding_dim
    batch_size = 2
    seq_len_q = 5
    seq_len_kv = 7
    embed_dim = 10
    q_input = torch.randn(batch_size,seq_len_q,embed_dim)
    k_input = torch.randn(batch_size,seq_len_q,embed_dim)
    # embedding_dim
    attn = Attention(embed_dim)

    out, weights = attn(q_input,k_input)
    # print(out)
    # print(weights)
    print(out.shape)
    print(weights.shape)
    print(torch.sum(weights, dim=1))