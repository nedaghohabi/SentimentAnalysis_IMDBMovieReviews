import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs, mask):
        # lstm_outputs: [batch_size, seq_len, hidden_dim]
        energy = self.attn(lstm_outputs).squeeze(-1)  # [batch, seq_len]
        energy = energy.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(energy, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs)  # [batch, 1, hidden_dim]
        return context.squeeze(1)  # [batch, hidden_dim]


class LSTMWithW2VEmbedding(nn.Module):
    def __init__(
        self,
        embedding_weights,
        freeze_embeddings,
        hidden_size,
        output_size,
        num_layers=1,
        bidirectional=False,
        dropout_rate=0.3,
        use_attention=False,  # new argument
    ):
        super(LSTMWithW2VEmbedding, self).__init__()
        n_vocab, embedding_dim = embedding_weights.shape

        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_weights), freeze=freeze_embeddings
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        if use_attention:
            self.attn = Attention(lstm_output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_output_dim, output_size)

    def forward(self, vector):
        x = vector.long()  # [batch, seq_len]
        mask = (x != 0)  # [batch, seq_len]

        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        outputs, (h_n, c_n) = self.lstm(embedded)  # [batch, seq_len, hidden_dim]

        if self.use_attention:
            context = self.attn(outputs, mask)  # [batch, hidden_dim]
        else:
            if self.bidirectional:
                context = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden_dim*2]
            else:
                context = h_n[-1]  # [batch, hidden_dim]

        out = self.dropout(context)
        out = self.fc(out)
        return {"sentiment": out}
