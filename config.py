import torch

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
eval_interval = 500
eval_iters = 200
n_embd = 384
num_layers = 6
num_heads = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('shakespere.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
