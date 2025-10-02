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

Download and place in the repo root, then run: `python main.py --mode generate --checkpoint path/to/model-improved-chat-Final-6.pkl`

## Setup
1. Clone the repo:
   git clone https://github.com/AudricHarris/Audric-Home-made-GPT
   cd AudricHarris/Audric-Home-made-GPT

Install dependencies:
   pip install -r requirements.txt

3. (Optional) For GPU training, ensure CUDA is installed.

## Usage
### Train the Model

- Trains for 10,000 iterations (configurable in `config.py`). this is an example do way more I would say at least 100,000 to 300,000 itterations
- Saves checkpoints every 100k steps and final model as `model-improved-chat-Final-6.pkl`.

### Generate Text / Chatbot
python main.py --mode generate --checkpoint model-improved-chat-Final-6.pkl

- Prompts for input; generates 100 tokens.
- Type 'exit' to quit.

## Customization
- Edit `config.py` for hyperparameters (e.g., `BLOCK_SIZE`, `N_LAYER`).
- Scale dataset: Increase `SAMPLE_SIZE` in `config.py` and adjust `MAX_ITERS`.
- For production: Add logging (e.g., via `logging` module) and evaluation metrics.

## Notes
- Training on full OpenWebText requires significant GPU resources (e.g., T4/A100).
- Generation uses top-k sampling for diversity.
- Tested on Python 3.11+ with PyTorch 2.0+.

## License
MIT License. Feel free to use/modify for your AI game!

If you encounter issues, open a GitHub issue.

---

## Français

Ce projet implémente un modèle de transformeur personnalisé de type GPT en utilisant PyTorch, affiné sur OpenWebText pour la génération de texte. Il est conçu pour un chatbot de jeu IA qui génère des réponses créatives. Le modèle est une implémentation from-scratch inspirée de nanoGPT, avec attention multi-tête, couches feed-forward et modélisation de langage causale.

## Fonctionnalités
- **Conception Modulaire** : Séparé en chargement de données, définition du modèle, entraînement et inférence.
- **Support GPU** : Détecte automatiquement CUDA/MPS/CPU.
- **Sauvegardes** : Reprise de l'entraînement à partir de fichiers pickle.
- **Chatbot Interactif** : Génération de jetons en temps réel avec échantillonnage par température.
- **Jeu de Données** : Streaming depuis OpenWebText (limité à 5000 échantillons pour la démo ; à scaler selon les besoins).

## Modèles Pré-entraînés
Utilisez des modèles pré-entraînés de Kaggle pour une inférence rapide :
- [Kaggle Models: audin-5-8](https://www.kaggle.com/models/wearwqlf/audin-5-8)
- Recommandé : `model-final-6` (meilleure cohérence et perplexité plus faible que les versions antérieures).

Téléchargez et placez dans la racine du repo, puis exécutez : `python main.py --mode generate --checkpoint path/to/model-final-6.pkl`

## Installation
1. Clonez le repo :
   git clone https://github.com/AudricHarris/Audric-Home-made-GPT
   cd AudricHarris/Audric-Home-made-GPT

2. Installez les dépendances :
   pip install -r requirements.txt

3. (Optionnel) Pour l'entraînement GPU, assurez-vous que CUDA est installé.

## Utilisation
### Entraîner le Modèle
python main.py --mode train

- Entraîne pour 10 000 itérations (configurable dans `config.py`). (Exemple mais en réalité c'est proche des 100k - 300k)
- Sauvegarde des points de contrôle tous les 100k pas et modèle final sous `model-improved-chat-Final-6.pkl`.

### Générer du Texte / Chatbot
python main.py --mode generate --checkpoint model-improved-chat-Final-6.pkl

- Demande une entrée ; génère 100 jetons.
- Tapez 'exit' pour quitter.

## Personnalisation
- Modifiez `config.py` pour les hyperparamètres (ex. : `BLOCK_SIZE`, `N_LAYER`).
- Échellez le jeu de données : Augmentez `SAMPLE_SIZE` dans `config.py` et ajustez `MAX_ITERS`.
- Pour la production : Ajoutez des logs (ex. : via module `logging`) et des métriques d'évaluation.

## Remarques
- L'entraînement sur OpenWebText complet nécessite des ressources GPU importantes (ex. : T4/A100).
- La génération utilise l'échantillonnage top-k pour la diversité.
- Testé sur Python 3.11+ avec PyTorch 2.0+.

## Licence
Licence MIT. N'hésitez pas à utiliser/modifier pour votre jeu IA !

En cas de problèmes, ouvrez une issue GitHub.
