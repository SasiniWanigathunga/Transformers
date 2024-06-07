from dataset import TextDataset
from torch.utils.data import DataLoader
import torch
from model import BigramLanguageModel
from config import chars, text, batch_size, max_iters, learning_rate, eval_interval, eval_iters, device

torch.manual_seed(1337)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

train_dataset = TextDataset(train_data)
val_dataset = TextDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# for iter in range(max_iters):
#   for xb, yb in train_dataloader:
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none = True)
#     loss.backward()
#     optimizer.step()
# print(loss.item())

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = dataset.get_batch(batch_size)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out

for iter in range(2):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = train_dataset.get_batch(batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx = context, max_new_tokens=1000)[0].tolist()))