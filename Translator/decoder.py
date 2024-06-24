import torch.nn as nn
from config import *
from utils import *


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(parameters_shape=([d_model]))
        self.dropout1 = nn.Dropout(p=dropout)
        self.cross_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNorm(parameters_shape=([d_model]))
        self.dropout2 = nn.Dropout(p=dropout)
        self.ffn = PositionWiseFeedforward(d_model=d_model, ffn_hidden_size=ffn_hidden_size, dropout=dropout)
        self.norm3 = LayerNorm(parameters_shape=([d_model]))
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, y, mask=None):
        # x : (30, 200, 512)
        # y : (30, 200, 512)
        # mask : (200, 200)

        out = self.self_attention(y, mask=mask) # (30, 200, 512)
        y = y + self.dropout1(out) # (30, 200, 512)
        y = self.norm1(y) # (30, 200, 512)

        out = self.cross_attention(y, x, mask=None) # (30, 200, 512)
        y = y + self.dropout2(out) # (30, 200, 512)
        y = self.norm2(y)  # (30, 200, 512)

        out = self.ffn(y) # (30, 200, 512)
        y = y + self.dropout3(out) # (30, 200, 512)
        y = self.norm3(y) # (30, 200, 512)

        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y
    

    

    
class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 ffn_hidden_size, 
                 dropout, 
                 num_layers, 
                 max_sequence_length, 
                 language_to_index, 
                 START_TOKEN, 
                 END_TOKEN, 
                 PAD_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
        self.decoder = SequentialDecoder(*[
            DecoderLayer(d_model, num_heads, ffn_hidden_size, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.decoder(x, y, self_attention_mask, cross_attention_mask)
        return y