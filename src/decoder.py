import torch.nn as nn

from multihead_attention import MultiHeadAttention
from feedforward import FeedForward


class DecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout=0.1):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):

        # Masked Self Attention
        attn1 = self.self_attention(x, x, x, mask)

        x = self.norm1(
            x + self.dropout(attn1)
        )

        # Encoder-Decoder Attention
        attn2 = self.cross_attention(
            x,
            encoder_output,
            encoder_output
        )

        x = self.norm2(
            x + self.dropout(attn2)
        )

        # Feed Forward
        ff = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout(ff)
        )

        return x