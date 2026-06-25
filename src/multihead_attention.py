import torch
import torch.nn as nn

from attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):

        batch_size = x.size(0)

        x = x.view(batch_size, -1, self.num_heads, self.d_k)

        return x.transpose(1, 2)

    def combine_heads(self, x):

        batch_size = x.size(0)

        x = x.transpose(1, 2).contiguous()

        return x.view(batch_size, -1, self.d_model)

    def forward(self, Q, K, V, mask=None):

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output, attention = self.attention(Q, K, V, mask)

        output = self.combine_heads(output)

        output = self.W_o(output)

        return output