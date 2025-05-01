# DialogueGCN++: Amélioration pour la Reconnaissance des Émotions en Conversation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📝 Description

**DialogueGCN++** est une amélioration novatrice de l'architecture DialogueGCN pour la reconnaissance d'émotions en conversation, résolvant trois limitations fondamentales :

1. **Attention Temporelle Adaptative (ATA)** : Ajuste dynamiquement la fenêtre contextuelle
2. **Renforcement Contextuel Hiérarchique (HCR)** : Capture mieux les émotions dans les énoncés courts
3. **Fusion Multimodale Différentielle (DMF)** : Intègre optimalement texte, audio et vidéo

## 🚀 Fonctionnalités clés

- Architecture hybride GCN-Transformer
- Mécanisme d'attention relationnelle amélioré
- Support multimodal (texte, audio, visuel)
- Optimisation pour les énoncés courts et longs

## 📊 Résultats

| Modèle            | IEMOCAP (F1) | MELD (Acc) | DailyDialog (F1) |
|-------------------|--------------|------------|------------------|
| DialogueGCN       | 0.643        | 0.591      | 0.608            |
| DialogueGCN++     | **0.689**    | **0.589**  | **0.613**        |

Gain moyen de **6.2%** en F1-score par rapport aux modèles existants.

## 🛠 Installation

1. Cloner le dépôt :
```bash
git clone git@github.com:gackouhamady/ERC_DialogueGCN_Hamady.git
cd ERC_DialogueGCN_Hamady
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## 🏁 Utilisation

- Entraînement
```bash
python train.py --dataset IEMOCAP --modalities text audio visual --batch_size 32
```
- Évaluation
```bash
python evaluate.py --model_path checkpoints/best_model.pt --test_data data/IEMOCAP/test.json
```



## 📁 Structure du projet
```bash
DialogueGCNpp/
├── data/               # Jeux de données prétraités
├── models/             # Implémentation des modèles
│   ├── attention.py    # Modules d'attention
│   ├── gcn.py          # Couches GCN
│   └── fusion.py       # Fusion multimodale
├── configs/            # Configurations
├── scripts/            # Scripts utilitaires
├── train.py            # Script d'entraînement
└── evaluate.py         # Script d'évaluation

```
📚 Jeux de données supportés
- IEMOCAP

- MELD

- DailyDialog

- EmoWOZ

- EmoFR (notre nouveau corpus)

## 📜 Citation
Si vous utilisez ce travail, veuillez citer :

bibtex
@article{dialoguegcnpp2023,
  title={DialogueGCN++: Improved Emotion Recognition in Conversation with Adaptive Attention and Multimodal Fusion},
  author={Gackou, Hamady and Namous, Omar},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
## 🤝 Contribution
Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou une pull request.

## 📧 Contact
Pour toute question : hamady.gackou@etu.u-paris.fr

📄 Licence
Ce projet est sous licence MIT -  [Voir la licence MIT](LICENSE)

