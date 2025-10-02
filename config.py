import torch

# Device setup
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Model hyperparameters
BATCH_SIZE = 2
BLOCK_SIZE = 512
VOCAB_SIZE = 50257  # GPT-2 vocab size
N_EMBD = 1048
N_HEAD = 16
N_LAYER = 24
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 10000
EVAL_ITERS = 25
TEMPERATURE = 0.8
TOP_K = 50
SAMPLE_SIZE = 5000  # Limit dataset for demo

# Paths
RESUME_CHECKPOINT = "/kaggle/input/audin-5-8/pytorch/default/5/model-improved-chat-Final-6.pkl"  # Update as needed
MODEL_SAVE_PATH = "model-improved-chat-Final-6.pkl"
