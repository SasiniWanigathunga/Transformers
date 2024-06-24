import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device

def scaled_dot_product_attention(query, key, value, mask=None):
    # q, k, v: (30, 8, 200, 64)
    d_k = query.shape[-1]
    wei = query @ key.transpose(-2, -1)  * d_k**(-0.5) # (30, 8, 200, 64) @ (30, 8, 64, 200) = (30, 8, 200, 200)
    if mask is not None:
        wei = wei.permute(1, 0, 2, 3) # (8, 30, 200, 200)
        wei += mask # (8, 30, 200, 200) + (200, 200) = (8, 30, 200, 200)
        wei = wei.permute(1, 0, 2, 3) # (30, 8, 200, 200)
    attention = F.softmax(wei, dim=-1) # (30, 8, 200, 200)
    values = attention @ value # (30, 8, 200, 200) @ (30, 8, 200, 64) = (30, 8, 200, 64)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float() # (256)
        denominator = torch.pow(10000, 2 * even_i / self.d_model) # (256)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)) # (200, 1)
        even_PE = torch.sin(position / denominator) # (200, 256)
        odd_PE = torch.cos(position / denominator) # (200, 256)
        stacked = torch.stack([even_PE, odd_PE], dim=2) # (200, 256, 2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2) # (200, 512)
        return PE
    
class SentenceEmbedding(nn.Module):
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.language_to_index = language_to_index
        self.vocab_size = len(language_to_index)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.device = device

    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(self.device)
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(self.device)
        x = self.dropout(x + pos)
        return x
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape # B: batch size, T: sequence length, C: d_model
        qkv = self.qkv_layer(x) # (30, 200, 3*512) = (30, 200, 1536)
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_size) # (30, 200, 8, 3*64) = (30, 200, 8, 192)
        qkv = qkv.permute(0, 2, 1, 3) # (30, 8, 200, 64)
        q, k, v = qkv.chunk(3, dim=-1)  # (30, 8, 200, 192/3) = (30, 8, 200, 64)
        values, attention = scaled_dot_product_attention(q, k, v, mask) # (30, 8, 200, 64)
        values = values.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_size) # (30, 200, 8, 64) -> (30, 200, 512)
        out = self.proj(values)
        return out
    
class LayerNorm(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # (512)
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # (512)

    def forward(self, x):
        dims = [-(i+1) for i in range(len(self.parameters_shape))] # [-1]
        mean = x.mean(dim=dims, keepdim=True) # (30, 200, 1)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True) # (30, 200, 1)
        std = (var + self.eps).sqrt() # (30, 200, 1)
        x = (x - mean) / std # (30, 200, 512)
        out = x * self.gamma + self.beta # (30, 200, 512) * (512) + (512) = (30, 200, 512)
        return out
    
class PositionWiseFeedforward(nn.Module):
    def __init__(self, d_model, ffn_hidden_size, dropout=0.1):
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_hidden_size, d_model)
        )

    def forward(self, x):
        return self.ffd(x) # x : (30, 200, 512) -> (30, 200, 2048) -> relu : (30, 200, 2048) -> (30, 200, 512) -> dropout : (30, 200, 512)
    
class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.q_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        B, T, C = x.shape # B: batch size, T: sequence length, C: d_model
        q = self.q_layer(y) # (30, 200, 512)
        kv = self.kv_layer(x) # (30, 200, 2*512) = (30, 200, 1024) # from encoder
        q = q.reshape(B, T, self.num_heads, self.head_size) # (30, 200, 8, 64)
        kv = kv.reshape(B, T, self.num_heads, 2 * self.head_size) # (30, 200, 8, 128)
        q = q.permute(0, 2, 1, 3) # (30, 8, 200, 64)
        kv = kv.permute(0, 2, 1, 3) # (30, 8, 200, 128)
        k, v = kv.chunk(2, dim=-1) # (30, 8, 200, 128/2) = (30, 8, 200, 64)
        values, attention = scaled_dot_product_attention(q, k, v, mask) # (30, 8, 200, 64)
        values = values.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_size) # (30, 200, 8, 64) -> (30, 200, 512)
        out = self.proj(values)
        return out
    