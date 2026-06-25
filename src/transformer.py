import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayer


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Separate embeddings for source and target
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final vocabulary projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):

        # Embeddings
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        # Positional Encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src)

        encoder_output = src

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, encoder_output)

        # Output layer
        output = self.fc_out(tgt)

        return output