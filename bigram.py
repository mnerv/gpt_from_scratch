import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how may independent sequences will we process in parallel
block_size = 8   # what is the maximum context length for predictions?
max_iters     = 3000
eval_interval = 300
learning_rate = 1e-2
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32  # number of embedded dimensions
# --------------------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encode: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decode: take a list of integeres, output a string

# encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# split up the data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    # move the data to the device because we might use the GPU
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.ln_head = nn.Linear(n_embed, vocab_size)  # Language model head

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor for integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C), batch by time by channel tensor
                                                   # In our case batch is 4, time is 8 and chanel is vocab size or 65
                                                   # Arrange it to (B, T, C) to logits which is the score for the next character in the sequence
        pos_emb = self.position_embedding_table(torch.arrange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        logits = self.ln_head(x)  # (B, T, C)  C and tok_emb C is not the same so (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

# create model and move it to the device
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

