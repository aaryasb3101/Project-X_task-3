import torch.nn as nn

from multihead_attention import MultiHeadAttention
from feedforward import FeedForward


class EncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout=0.1):

        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn = self.attention(x,x,x)

        x = self.norm1(
            x + self.dropout(attn)
        )

        ff = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(ff)
        )

        return x