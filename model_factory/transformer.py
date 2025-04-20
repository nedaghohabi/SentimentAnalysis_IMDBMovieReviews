import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerWithW2VEmbedding(nn.Module):
    def __init__(
        self,
        embedding_weights,
        freeze_embeddings,
        hidden_size,
        output_size,
        num_layers=2,
        num_heads=4,
        dim_feedforward=512,
        dropout_rate=0.1,
        max_seq_len=1000,
    ):
        super().__init__()
        n_vocab, embedding_dim = embedding_weights.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_weights), freeze=freeze_embeddings
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, output_size)

    def forward(self, vector):
        x = vector.long()  # [batch_size, seq_len]
        mask = (x == 0)  # padding mask

        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Mean pooling (ignoring padding)
        mask_inv = (~mask).unsqueeze(-1).float()
        x = (x * mask_inv).sum(dim=1) / mask_inv.sum(dim=1).clamp(min=1e-8)

        x = self.dropout(x)
        x = self.fc(x)

        return {"sentiment": x}
