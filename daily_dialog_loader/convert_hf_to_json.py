import os
import json
from datasets import load_dataset

# Charger le dataset HuggingFace (il pointe vers les fichiers .arrow automatiquement)
dataset = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True)

# Dossier de sortie
os.makedirs("dailydialog_json", exist_ok=True)

def convert_split(split_name, data):
    output = []
    for sample in data:
        dialogue = []
        for utt, act, emo in zip(sample['dialog'], sample['act'], sample['emotion']):
            dialogue.append({
                "text": utt,
                "act": act,
                "emotion": emo
            })
        output.append({"dialogue": dialogue})
    
    with open(f"dailydialog_json/{split_name}_1.json", "w", encoding="utf-8") as f:
        for d in output:
            json.dump(d, f)
            f.write('\n')

# Convertir chaque split
convert_split("train", dataset["train"])
convert_split("dev", dataset["validation"])
convert_split("test", dataset["test"])

print("✅ Conversion terminée. Fichiers enregistrés dans le dossier 'dailydialog_json/'")
