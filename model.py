import torch
import torch.nn as nn
from torch.nn import functional as F
from config import vocab_size, n_embd, block_size, device


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        # compute the attention scores
        wei = q @ k.transpose(-2, -1)  * C**(-0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        # perform weighted aggregation of values
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,T) @ (B,T,head_size) = (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(num_heads = 4, head_size = n_embd//4) # 4 heads of 32//4 = 8 dimensional self attention , then concatenate to get size 32 = n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_embd = self.position_embedding_table(torch.arange(idx.shape[1], device = device))  # (T,n_embd)
        x = tok_embd + pos_embd  # (B,T,n_embd)
        x = self.sa_heads(x)  # apply one head of self attention # (B,T,n_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B,C)
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B,T+1)
        return idx
