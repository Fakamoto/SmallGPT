import torch
import os
from model import SmallGPTLanguageModel, get_batch, get_loss, estimate_loss

# Hyperparameters
batch_size = 128
block_size = 32
max_iters = 10
print_progress_iters = 10
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses[True]:.4f}, val loss {losses[False]:.4f}")

        xb, yb = get_batch(training=True)

        logits = model(xb)
        loss = get_loss(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(model)
    print(f"step {max_iters}: train loss {losses[True]:.4f}, val loss {losses[False]:.4f}")
    print(model.speak())
    torch.save(model.state_dict(), 'darwin_model.pth')
