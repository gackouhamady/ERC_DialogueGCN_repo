# Re-evaluating DialogueGCN for Emotion Recognition in Conversation


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📘 Project Overview

This project conducts a **critical reproduction and evaluation** of the DialogueGCN model — a graph-based neural network designed for **Emotion Recognition in Conversations (ERC)**. Our work involves faithful reimplementation, empirical validation on benchmark datasets, and exploration of its practical and architectural limitations.

## 🎯 Objectives

- Reproduce the original DialogueGCN architecture and training setup.
- Evaluate its performance on **IEMOCAP**, **MELD**, and **DailyDialog** datasets.
- Analyze the effects of key hyperparameters (dropout, learning rate, context window).
- Test the model's **scalability** and **hardware sensitivity**, especially on **EmoryNLP** (CPU vs GPU).
- Provide clear insights into **graph-based architectures** for conversational emotion analysis.

## 🧠 Why DialogueGCN?

DialogueGCN (Ghosal et al., EMNLP-IJCNLP 2019) models dialogues as relational graphs to capture both **temporal** and **inter-speaker dependencies**, using Relational GCN layers. Its design allows for fine-grained modeling of interpersonal emotional dynamics, surpassing sequential baselines (LSTM, GRU, DialogueRNN) in accuracy and expressivity.

## 🔬 Key Results

```bash

| Dataset      | F1-score (original) | F1-score (reproduced) | Notes                                           |
|--------------|---------------------|------------------------|------------------------------------------------|
| IEMOCAP      | 64.18%              | 63.9%                  | Stable with weighted loss + dropout            |
| DailyDialog  | N/A                 | 82.28%                 | Excellent generalization, clean structure      |
| MELD         | 58.10%              | 48.12%                 | Convergence issues, unbalanced label problem   |
| EmoryNLP     | --                  | --                     | Failed training on CPU (memory bottleneck)     |
```
## ⚙️ Technologies

- Python 3.10
- PyTorch 2.7.0
- PyTorch Geometric 2.6.1
- GloVe embeddings (300d)
- TensorBoard, Pandas, scikit-learn

## 📂 Project Structure









## 📁 Structure du projet
```bash
ERC_DIALOGUEGCN_HAMADY/
├── .vscode/                      # Configurations VSCode
├── data/                         # Datasets et fichiers prétraités
├── glove/                        # Fichiers d'embedding GloVe
├── models/                       # Implémentations de modèles (DialogueGCN, etc.)
├── report/                       # Rapport scientifique (PDF, .tex, etc.)
├── scripts/                      # Scripts d'entraînement par dataset
│   ├── saved/                    # Checkpoints sauvegardés
│   ├── train_evaluate_dailydialogue.py
│   ├── train_evaluate_emory.py
│   ├── train_evaluate_iemocap.py
│   └── train_evaluate_meld.py
├── utils/                        # Utilitaires Python
│   ├── __pycache__/
│   ├── daily_dialog_loader/     # Loader spécifique pour DailyDialog
│   └── video_presentation/      # Éléments pour la vidéo de présentation
├── .gitignore                    # Fichiers ignorés par Git
├── discussion.ouverte.txt       # Notes ouvertes de discussion ou TODO
├── LICENSE                       # Licence du projet
├── preprocess_dailydialog.py    # Script de prétraitement pour DailyDialog
├── preprocess_emorynlp.py       # Script de prétraitement pour EmoryNLP
├── preprocess_meld.py           # Script de prétraitement pour MELD
├── README.md                    # Présentation du projet
├── requirements.txt             # Dépendances Python
├── results_resume.txt           # Résumé synthétique des résultats
└── results.txt                  # Résultats détaillés


```

## 🧪 Reproducibility

All hyperparameters, Scripts, and model checkpoints are provided. To replicate the main results

##🔍 Highlights
- Graph-based architectures significantly outperform sequential models in structured ERC.

- Training fails on CPU for EmoryNLP due to O(N²) memory cost of relational GCN.

- Small batch sizes and dot-product attention yield the best F1 scores.

- Class weighting crucial for imbalanced datasets like MELD.

📈 Future Directions
- Integrate multimodal features (audio, video) for richer emotion cues.

- Explore graph sparsification for efficient CPU training.

- Apply explainability techniques (e.g., GNNExplainer).

- Investigate transfer learning to social media or low-resource domains.

## 📚 References Clés
- Ghosal et al., DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation, EMNLP-IJCNLP 2019.

- Majumder et al., DialogueRNN, AAAI 2019.

- Poria et al., MELD Dataset, ACL 2019.

- Schlichtkrull et al., Relational GCN, ESWC 2018.

#### Author: Hamady GACKOU
#### Supervisor: Dr. Séverine AFFELT, Université Paris Cité
## 📅 Project Year: 2025
## 📄 See full report in /report/







## 📄 Licence
[See](LICENSE)

