# GPT Transformer for AI Game / Transformeur GPT pour Jeu IA

## English

This project implements a custom GPT-like transformer model using PyTorch, fine-tuned on OpenWebText for text generation. It's designed for an AI game chatbot that generates creative responses. The model is a from-scratch implementation inspired by nanoGPT, with multi-head attention, feed-forward layers, and causal language modeling.

## Features
- **Modular Design**: Separated into data loading, model definition, training, and inference.
- **GPU Support**: Automatically detects CUDA/MPS/CPU.
- **Checkpointing**: Resume training from pickle files.
- **Interactive Chatbot**: Real-time token generation with temperature sampling.
- **Dataset**: Streaming from OpenWebText (limited to 5000 samples for demo; scale up as needed).

## Pre-trained Models
Use pre-trained models from Kaggle for quick inference:
- [Kaggle Models: audin-5-8](https://www.kaggle.com/models/wearwqlf/audin-5-8)
- Recommended: `model-final-6` (better coherence and lower perplexity than earlier versions).

Download and place in the repo root, then run: `python main.py --mode generate --checkpoint path/to/model-final-6.pkl`

## Setup
1. Clone the repo:
   git clone https://github.com/AudricHarris/Audric-Home-made-GPT
   cd AudricHarris/Audric-Home-made-GPT
