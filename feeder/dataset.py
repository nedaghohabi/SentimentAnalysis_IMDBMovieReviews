import torch
from torch.utils.data import Dataset


class SentimentAnalysisDataset(Dataset):
    def __init__(self, vectors, targets):
        """
        Args:
            doc_vectors (list or np.array): List or array of document vectors.
            targets (list or np.array): List or array of sentiment targets.
        """
        self.vectors = vectors
        self.targets = targets

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        sample_vector = torch.tensor(self.vectors[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        output = {"inputs": {"vector": sample_vector},
                  "labels": {"sentiment": target}}
        return output
