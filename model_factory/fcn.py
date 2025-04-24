import torch

import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vector):
        x = vector
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return {"sentiment": x}
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNWithW2VEmbedding(nn.Module):
    def __init__(self, embedding_weights, freeze_embeddings, aggregator, hidden_size, output_size, dropout_rate=0.3):
        super(FCNWithW2VEmbedding, self).__init__()
        n_vocab, embedding_dim = embedding_weights.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_weights), freeze=freeze_embeddings
        )
        self.aggregator = aggregator.lower()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, vector):
        x = vector.long()  # [batch_size, seq_len]
        mask = (x != 0).float()  # [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        if self.aggregator == "mean":
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        elif self.aggregator == "sum":
            x = (x * mask.unsqueeze(-1)).sum(dim=1)
        elif self.aggregator == "max":
            x[mask == 0] = float('-inf')
            x, _ = x.max(dim=1)
            x[x == float('-inf')] = 0  # replace all -inf back to 0
        elif self.aggregator == "min":
            x[mask == 0] = float('inf')
            x, _ = x.min(dim=1)
            x[x == float('inf')] = 0
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return {"sentiment": x}
