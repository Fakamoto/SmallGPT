import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 1000
eval_iters = 50
print_progress_iters = 10
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 256
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# create a mapping from characters to integers
character_to_token: dict[str, int] = {character: i for i, character in enumerate(vocab)}
token_to_character: dict[int, str] = {i: character for i, character in enumerate(vocab)}

def encode(text: str) -> list[int]:
    return [character_to_token[character] for character in text]

def decode(tokens: list[int]) -> str:
    if isinstance(tokens, torch.Tensor):
        tokens.squeeze(0)
        tokens = tokens.tolist()
    return "".join([token_to_character[token] for token in tokens])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
data_split = int(len(data) * 0.9) # first 90% will be train, rest val
train_data = data[:data_split]
val_data = data[data_split:]

def get_batch(training: bool = False, block_size = block_size, batch_size = batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if training else val_data
    starting_indexes_of_batches = torch.randint(len(data) - block_size, (batch_size,))

    inputs = []
    targets = []

    for index in starting_indexes_of_batches:
        input_sequence = data[index: index + block_size]
        inputs.append(input_sequence)

        target_sequence = data[index + 1: index + block_size + 1]
        targets.append(target_sequence)
        
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

def get_loss(logits, target) -> float:
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    target = target.view(B*T)
    loss = F.cross_entropy(logits, target)
    return loss

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for training in [True, False]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            inputs_batches, targets_batches = get_batch(training=training)
            logits: torch.Tensor = model(inputs_batches)
            loss = get_loss(logits, targets_batches)
            losses[i] = loss.item()
        out[training] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SmallGPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    def speak(self, max_new_tokens=300):
        return decode(self.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=max_new_tokens)[0].tolist())

    def generate_from_text(self, text: str, max_new_tokens=300):
        idx = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)
        return decode(self.generate(idx, max_new_tokens=max_new_tokens)[0].tolist())


if __name__ == '__main__':
    model = SmallGPTLanguageModel()
    
    if os.path.exists('darwin_model.pth'):
        model.load_state_dict(torch.load('darwin_model.pth'))
        print("Loaded model weights from darwin_model.pth")
    else:
        print("Training from scratch")
    
    model = model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):

        if iter % print_progress_iters == 0:
            train_loss, val_loss = estimate_loss(model)
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        xb, yb = get_batch(training=True)

        logits = model(xb)
        loss = get_loss(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    train_loss, val_loss = estimate_loss(model)
    print(f"step {max_iters}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
    print(model.speak())
    torch.save(model.state_dict(), 'darwin_model.pth')