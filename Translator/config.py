import torch

d_model = 512
num_heads = 8
dropout = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden_size = 2048
num_layers = 5
sinhala_vocab_size = 10000
english_vocab_size = 10000
START_TOKEN = 0
END_TOKEN = 1
PAD_TOKEN = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   