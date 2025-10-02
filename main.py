import argparse
from data import load_dataset
from train import train_model
from generate import chatbot, load_model
from config import MODEL_SAVE_PATH

def main():
    parser = argparse.ArgumentParser(description="GPT Transformer for AI Game")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--checkpoint', type=str, default=MODEL_SAVE_PATH, help="Path to model checkpoint for generation")
    args = parser.parse_args()

    tokenizer = None
    train_dataset = None
    val_dataset = None

    if args.mode == 'train':
        tokenizer = load_tokenizer()
        train_dataset, val_dataset, _ = load_dataset(tokenizer)
        model = train_model(tokenizer, train_dataset, val_dataset)
    elif args.mode == 'generate':
        tokenizer = load_tokenizer()
        _, _, tokenizer = load_dataset(tokenizer)  # Load tokenizer only
        model = load_model(args.checkpoint)
        if model:
            chatbot(model, tokenizer)
        else:
            print("Failed to load model. Exiting.")

if __name__ == "__main__":
    main()
