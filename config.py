import torch

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3
eval_interval = 300
eval_iters = 200
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('shakespere.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
