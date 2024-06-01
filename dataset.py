from torch.utils.data import Dataset
import torch
from config import block_size, device


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+block_size]
        y = self.data[idx+1:idx+block_size+1]
        x, y = x.to(device), y.to(device)
        return x, y
    
    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data)-block_size, (batch_size,))
        x = torch.stack([self.data[i:i+block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    