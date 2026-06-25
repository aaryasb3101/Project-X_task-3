import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):

        d_k = Q.size(-1)

        # QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale
        scores = scores / math.sqrt(d_k)

        # Optional mask (used in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Multiply by values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights