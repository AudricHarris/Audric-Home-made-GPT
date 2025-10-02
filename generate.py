import torch
import sys
from data import encode, decode, load_tokenizer
from model import GPTLanguageModel
from config import VOCAB_SIZE, device, TEMPERATURE, TOP_K, BLOCK_SIZE, MODEL_SAVE_PATH

def chatbot(model, tokenizer):
    """Interactive chatbot loop."""
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        try:
            prompt = input("Enter a prompt: ").strip()
            if prompt.lower() == 'exit':
                break
            if not prompt:
                continue

            context = torch.tensor([encode(prompt, tokenizer)], dtype=torch.long, device=device)
            generated_tokens = model.generate(context, max_new_tokens=100, temperature=TEMPERATURE, top_k=TOP_K, tokenizer=tokenizer)[0].tolist()
            generated_text = decode(generated_tokens, tokenizer)
            print("\nFull generated text:")
            print(generated_text)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def load_model(checkpoint_path=MODEL_SAVE_PATH):
    """Load model from pickle."""
    try:
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        model = model.to(device)
        print(f"Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
