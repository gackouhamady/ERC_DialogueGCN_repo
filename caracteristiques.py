import pickle
import numpy as np

fichier_pkl = 'data/MELD_features/MELD_features.pkl'

with open(fichier_pkl, 'rb') as f:
    data = pickle.load(f)

# Configuration
split_name = "train"  # ou "val", "test"
dialogue_id = "dia0"
utter_idx = 0

split_map = {
    "train": (0, 1, 2),
    "val":   (3, 4, 5),
    "test":  (6, 7, 8),
}
utt_idx, feat_idx, lab_idx = split_map[split_name]

utterance_ids = data[utt_idx][dialogue_id]
utterance_id = utterance_ids[utter_idx]

print(f"\n🆔 Utterance ID (from utterances): {utterance_id}")

# Test de la structure de data[feat_idx]
example_keys = list(data[feat_idx].keys())[:5]
print(f"🔎 Exemple de clés dans data[{feat_idx}] : {example_keys}")

if utterance_id in data[feat_idx]:
    # Structure : dict[utterance_id]
    features = data[feat_idx][utterance_id]
    label = data[lab_idx][utterance_id]
elif dialogue_id in data[feat_idx]:
    # Structure : dict[dialogue_id] → list
    features = data[feat_idx][dialogue_id][utter_idx]
    label = data[lab_idx][dialogue_id][utter_idx]
else:
    raise KeyError(f"Impossible de trouver '{utterance_id}' ou '{dialogue_id}' dans data[{feat_idx}]")

# Affichage
print(f"\n🗣 Utterance : {utterance_id}")
print(f"🎯 Emotion label : {label}")
print("\n🔧 Features :")
for k, v in features.items():
    print(f"  - '{k}': type={type(v)}, shape={np.array(v).shape}")
