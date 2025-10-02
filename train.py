import torch
import pickle
import os
from tqdm import tqdm
from data import get_batch, load_dataset
from model import GPTLanguageModel
from config import (
    RESUME_CHECKPOINT, MODEL_SAVE_PATH, MAX_ITERS, EVAL_ITERS, LEARNING_RATE, device, VOCAB_SIZE
)

def estimate_loss(model, train_dataset, val_dataset):
    """Estimate loss on train/val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_dataset, val_dataset)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(tokenizer, train_dataset, val_dataset):
    """Train the model, with optional checkpoint resume."""
    print("Initializing model...")
    model = GPTLanguageModel(VOCAB_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load checkpoint if exists
    start_iter = 0
    best_val_loss = float('inf')
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading model from {RESUME_CHECKPOINT}...")
        try:
            with open(RESUME_CHECKPOINT, 'rb') as f:
                checkpoint = pickle.load(f)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_iter = checkpoint.get('iteration', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print("Starting training from scratch...")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=LEARNING_RATE / 10)

    # Resume optimizer if checkpoint has it
    if start_iter > 0 and 'optimizer_state_dict' in locals().get('checkpoint', {}):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Starting from iteration: {start_iter}")

    # Training loop
    print("Starting training...")
    for iter in tqdm(range(start_iter, MAX_ITERS), desc="Training"):
        if iter % EVAL_ITERS == 0:
            losses = estimate_loss(model, train_dataset, val_dataset)
            print(f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

            if iter > 0 and iter % 100000 == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iter,
                    'best_val_loss': min(best_val_loss, losses['val']),
                    'losses': losses
                }
                torch.save(checkpoint, f'checkpoint_{iter}.pt')
                print(f"Checkpoint saved at iteration {iter}")
                del checkpoint
                torch.cuda.empty_cache()

        xb, yb = get_batch('train', train_dataset, val_dataset)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        del xb, yb, logits, loss
        torch.cuda.empty_cache()

    # Save final model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return model
