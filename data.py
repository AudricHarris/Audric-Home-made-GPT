import torch
import random
from itertools import islice
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from config import SAMPLE_SIZE, BLOCK_SIZE, BATCH_SIZE, device

def load_tokenizer():
    """Load GPT-2 tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"End token: {tokenizer.pad_token}")
    return tokenizer

def load_dataset(tokenizer):
    """Load and prepare streaming dataset, split into train/val, and tokenize."""
    print("Loading streaming dataset...")
    raw_dataset = load_dataset("vietgpt/openwebtext_en", split="train", streaming=True)
    raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10000)
    raw_dataset = islice(raw_dataset, SAMPLE_SIZE)

    train_list, val_list = [], []
    for i, sample in enumerate(tqdm(raw_dataset, desc="Splitting dataset")):
        if i % 10 == 0:  # 10% validation
            val_list.append(sample)
        else:
            train_list.append(sample)

    print(f"Train samples: {len(train_list)}, Val samples: {len(val_list)}")

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding=False)

    train_dataset = [tokenize_function(s) for s in train_list]
    val_dataset = [tokenize_function(s) for s in val_list]
    return train_dataset, val_dataset, tokenizer

def encode(text, tokenizer, max_length=1024):
    """Encode text to tokens."""
    return tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)

def decode(tokens, tokenizer):
    """Decode tokens to text."""
    return tokenizer.decode(tokens, skip_special_tokens=True)

def get_random_chunk(split, train_dataset, val_dataset):
    """Get a random chunk of tokens from the dataset."""
    dataset = train_dataset if split == 'train' else val_dataset
    while True:
        sample = random.choice(dataset)['input_ids']
        if len(sample) >= BLOCK_SIZE:
            break
    start = random.randint(0, len(sample) - BLOCK_SIZE)
    chunk = sample[start: start + BLOCK_SIZE]
    return torch.tensor(chunk, dtype=torch.long)

def get_batch(split, train_dataset, val_dataset):
    """Generate a batch of input/target tensors."""
    data = get_random_chunk(split, train_dataset, val_dataset)
    if len(data) < BLOCK_SIZE * BATCH_SIZE:
        needed_length = BLOCK_SIZE * BATCH_SIZE + 1
        data = data.repeat((needed_length // len(data)) + 1)[:needed_length]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
