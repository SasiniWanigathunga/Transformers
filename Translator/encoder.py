import torch.nn as nn
from config import *
from utils import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(parameters_shape=([d_model]))
        self.dropout1 = nn.Dropout(p=dropout)
        self.ffn = PositionWiseFeedforward(d_model=d_model, ffn_hidden_size=ffn_hidden_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(parameters_shape=([d_model]))
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        residual_x = x # (30, 200, 512)
        x = self.attention(x, mask=None) # (30, 200, 512)
        x = self.dropout1(x) # (30, 200, 512)
        x = self.norm1(x + residual_x) # (30, 200, 512)
        residual_x = x # (30, 200, 512)
        x = self.ffn(x) # (30, 200, 512)
        x = self.dropout2(x) # (30, 200, 512)
        x = self.norm2(x + residual_x) # (30, 200, 512)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 ffn_hidden_size, 
                 num_layers, 
                 dropout, 
                 max_sequence_length, 
                 language_to_index, 
                 START_TOKEN, 
                 END_TOKEN, 
                 PAD_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
        self.encoder = SequentialEncoder(*[
            EncoderLayer(d_model, num_heads, ffn_hidden_size, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.encoder(x, self_attention_mask) # just the padding mask
        return x
