import torch.nn as nn
from config import *
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 ffn_hidden_size, 
                 num_layers, 
                 dropout, 
                 max_sequence_length, 
                 sinhala_vocab_size, 
                 english_vocab_size, 
                 sinhala_to_index, 
                 english_to_index, 
                 START_TOKEN, 
                 END_TOKEN, 
                 PAD_TOKEN, 
                 device):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, ffn_hidden_size, num_layers, dropout, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
        self.decoder = Decoder(d_model, num_heads, ffn_hidden_size, dropout, num_layers, max_sequence_length, sinhala_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
        self.linear = nn.Linear(d_model, sinhala_vocab_size)
        self.device = device

    def forward(self, x, y, encoder_self_attention_mask, decoder_self_attention_mask, cross_attention_mask, encoder_start_token, encoder_end_token, decoder_start_token, decoder_end_token):
        x = self.encoder(x, encoder_self_attention_mask, start_token=encoder_start_token, end_token=encoder_end_token)
        y = self.decoder(y, x, decoder_self_attention_mask, cross_attention_mask, start_token=decoder_start_token, end_token=decoder_end_token)
        out = self.linear(y)
        return out