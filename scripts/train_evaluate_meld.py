import os
import sys

 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
 
from models.baseline_meld import BaselineMELD  # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-classify", choices=["Emotion", "Sentiment"], required=True)
    parser.add_argument("-modality", choices=["text", "audio", "bimodal"], required=True)
    parser.add_argument("-train", action="store_true", help="Lancer l'entraînement")
    parser.add_argument("-test", action="store_true", help="Lancer le test")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques d'entraînement")

    args = parser.parse_args()

    os.makedirs("../data/MELD_features", exist_ok=True)
    os.makedirs("../data/models", exist_ok=True)
    os.makedirs("../data/embeddings", exist_ok=True)

    model = BaselineMELD(args) # type: ignore
    model.load_data()
    if args.train:
        model.train_model()
    elif args.test:
        model.test_model()
    else:
        print("Spécifiez -train ou -test.")