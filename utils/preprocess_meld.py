import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import pickle
from collections import defaultdict
from tqdm import tqdm # type: ignore

# === Configuration ===
data_dir = "./data/MELD_csv"  # dossier contenant les fichiers csv
save_path = "./data/MELD_features/MELD_features.pkl"
embedding_dim = 300  # dimension des vecteurs simulés

# === Encodage émotion/sentiment ===
emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3,
               'joy': 4, 'disgust': 5, 'anger': 6}
sentiment2idx = {'positive': 2, 'neutral': 1, 'negative': 0}

# === Fonctions auxiliaires ===
def dummy_embed(text):
    """Génère un vecteur aléatoire pour simuler l'embedding du texte."""
    tokens = text.strip().split()
    return [np.random.rand(embedding_dim).astype(np.float32) for _ in tokens]

# === Dictionnaires à remplir ===
videoIDs = defaultdict(list)
videoSpeakers = defaultdict(list)
videoLabelsEmotion = defaultdict(list)
videoLabelsSentiment = defaultdict(list)
videoText = defaultdict(list)
videoAudio = defaultdict(list)  # dummy audio
videoSentence = defaultdict(list)
trainVid, testVid = [], []

def process_file(filename, split_name):
    df = pd.read_csv(filename)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        dia_id = f"{row['Dialogue_ID']}_{row['Utterance_ID']}"
        dialog_key = f"dia{row['Dialogue_ID']}"

        # embeddings factices
        text = str(row["Utterance"])
        text_embed = dummy_embed(text)

        videoIDs[dialog_key].append(dia_id)
        videoSpeakers[dialog_key].append(row["Speaker"])
        videoSentence[dialog_key].append(text)
        videoText[dialog_key].append(text_embed)
        videoAudio[dialog_key].append(np.random.rand(len(text_embed), embedding_dim))  # dummy audio

        # labels
        emo_label = row["Emotion"]
        sent_label = row["Sentiment"]
        videoLabelsEmotion[dialog_key].append(emotion2idx.get(emo_label, 0))
        videoLabelsSentiment[dialog_key].append(sentiment2idx.get(sent_label, 1))

        if split_name == "train":
            if dialog_key not in trainVid:
                trainVid.append(dialog_key)
        else:
            if dialog_key not in testVid:
                testVid.append(dialog_key)

# === Traiter les fichiers CSV ===
process_file(os.path.join(data_dir, "train_sent_emo.csv"), "train")
process_file(os.path.join(data_dir, "dev_sent_emo.csv"), "test")
process_file(os.path.join(data_dir, "test_sent_emo.csv"), "test")

# === Sauvegarder dans un fichier .pkl ===
with open(save_path, "wb") as f:
    pickle.dump((
        dict(videoIDs),
        dict(videoSpeakers),
        dict(videoLabelsEmotion),
        dict(videoText),
        dict(videoAudio),
        dict(videoSentence),
        trainVid,
        testVid,
        dict(videoLabelsSentiment)
    ), f)

print(f"\n✅ Sauvegardé sous : {save_path}")
