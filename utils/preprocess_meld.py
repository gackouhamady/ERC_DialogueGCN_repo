import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore

# Support TensorFlow ≥2.13
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
except ImportError:
    from keras_preprocessing.sequence import pad_sequences # type: ignore


class MELDDataPreprocessor:
    """
    Classe pour prétraiter les données MELD et les préparer pour le modèle bc_LSTM
    """
    
    def __init__(self, data_dir="../data/MELD_csv", embedding_dim=300, max_sent_length=50):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))
        self.embedding_dim = embedding_dim
        self.max_sent_length = max_sent_length
        self.emotion2idx = {
            'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3,
            'joy': 4, 'disgust': 5, 'anger': 6
        }
        self.sentiment2idx = {'positive': 2, 'neutral': 1, 'negative': 0}
        
    def load_glove_embeddings(self, glove_path):
        """Charge les embeddings GloVe pré-entraînés"""
        glove_path = os.path.abspath(os.path.join(os.path.dirname(__file__), glove_path))
        print(f"Loading GloVe embeddings from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
    
    def create_embedding_matrix(self, word_index, embeddings_index):
        """Crée la matrice d'embedding pour notre vocabulaire"""
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        for word, i in word_index.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        return embedding_matrix
    
    def process_dataset(self):
        """Traite l'ensemble du dataset MELD"""
        print("Processing MELD dataset...")
        
        # Vérification de l'existence des fichiers
        train_path = os.path.join(self.data_dir, "train_sent_emo.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Fichier train_sent_emo.csv introuvable dans {self.data_dir}")
        
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(os.path.join(self.data_dir, "dev_sent_emo.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test_sent_emo.csv"))
        
        # Traitement du texte
        all_text = pd.concat([train_df['Utterance'], dev_df['Utterance'], test_df['Utterance']])
        
        vocab = set()
        word_counts = Counter()
        for text in all_text:
            tokens = str(text).strip().split()
            vocab.update(tokens)
            word_counts.update(tokens)
        
        self.vocab = vocab
        self.word_index = {w: i+1 for i, w in enumerate(vocab)}  # 0 pour le padding
        
        # Traitement des splits
        train_data = self._process_split(train_df, "train")
        val_data = self._process_split(dev_df, "val")
        test_data = self._process_split(test_df, "test")
        
        # Calcul de la longueur maximale des dialogues
        self.max_utts = max(
            max(len(d) for d in train_data['dialogue_ids'].values()),
            max(len(d) for d in val_data['dialogue_ids'].values()),
            max(len(d) for d in test_data['dialogue_ids'].values())
        )
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'word_index': self.word_index,
            'vocab': self.vocab,
            'emotion_labels': self.emotion2idx,
            'sentiment_labels': self.sentiment2idx
        }
    
    def _process_split(self, df, split_name):
        """Traite un split particulier du dataset"""
        print(f"Processing {split_name} split...")
        
        data = {
            'text': [],
            'dialogue_ids': defaultdict(list),
            'speakers': defaultdict(list),
            'emotions': [],
            'sentiments': [],
            'utterance_ids': []
        }
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            dia_id = f"dia{row['Dialogue_ID']}"
            utt_id = f"{dia_id}_{row['Utterance_ID']}"
            
            tokens = str(row["Utterance"]).split()
            idxs = [self.word_index.get(t, 0) for t in tokens]
            padded = pad_sequences([idxs], maxlen=self.max_sent_length, padding='post')[0]
            
            data['text'].append(padded)
            data['dialogue_ids'][dia_id].append(row['Utterance_ID'])
            data['speakers'][dia_id].append(row['Speaker'])
            data['utterance_ids'].append(utt_id)
            
            data['emotions'].append(self.emotion2idx.get(row['Emotion'], 0))
            data['sentiments'].append(self.sentiment2idx.get(row['Sentiment'], 1))
        
        return data
    
    def prepare_for_bc_lstm(self, processed_data, mode='emotion'):
        """
        Prépare les données pour le modèle bc_LSTM
        mode: 'emotion' ou 'sentiment'
        """
        print(f"Preparing data for bc_LSTM ({mode})...")
        
        key = 'emotions' if mode == 'emotion' else 'sentiments'
        
        # Chemin absolu pour les embeddings GloVe
        glove_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../glove/glove.6B.300d.txt"))
        
        return {
            'train': self._prepare_split(processed_data['train'], key),
            'val': self._prepare_split(processed_data['val'], key),
            'test': self._prepare_split(processed_data['test'], key),
            'W': self.create_embedding_matrix(self.word_index, self.load_glove_embeddings(glove_path))
        }
    
    def _prepare_split(self, split_data, label_key):
        """Prépare un split pour bc_LSTM"""
        dialogues, labels, ids, lengths = [], [], {}, []
        max_utts = self.max_utts
        
        for dia_id, utts in split_data['dialogue_ids'].items():
            feats, labs = [], []
            for u in sorted(utts):
                idx = split_data['utterance_ids'].index(f"{dia_id}_{u}")
                feats.append(split_data['text'][idx])
                labs.append(split_data[label_key][idx])
            
            n = len(feats)
            pad_feats = np.zeros((max_utts, self.max_sent_length))
            pad_feats[:n] = feats
            
            num_classes = len(self.emotion2idx) if label_key == 'emotions' else len(self.sentiment2idx)
            onehot = np.zeros((max_utts, num_classes))
            onehot[np.arange(n), labs] = 1
            
            dialogues.append(pad_feats)
            labels.append(onehot)
            ids[dia_id] = utts
            lengths.append(n)
        
        mask = np.zeros((len(dialogues), max_utts), dtype='float32')
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        
        return {
            'dialogue_features': np.array(dialogues),
            'dialogue_labels': np.array(labels),
            'dialogue_ids': ids,
            'dialogue_lengths': lengths,
            'mask': mask
        }
    
    def save_processed_data(self, data, output_path):
        """Sauvegarde les données prétraitées"""
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Processed data saved to {output_path}")


def main():
    # Chemins absolus pour les répertoires
    pickles_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/MELD_features"))
    meld_csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/MELD_csv"))
    
    # Création des répertoires nécessaires
    os.makedirs(pickles_dir, exist_ok=True)
    os.makedirs(meld_csv_dir, exist_ok=True)
    
    # Initialisation du prétraitement
    preprocessor = MELDDataPreprocessor(data_dir=meld_csv_dir)
    
    try:
        # Traitement du dataset
        processed_data = preprocessor.process_dataset()
        
        # Préparation et sauvegarde des données pour Emotion
        emotion_data = preprocessor.prepare_for_bc_lstm(processed_data, mode='emotion')
        preprocessor.save_processed_data(emotion_data, os.path.join(pickles_dir, "MELD_emotion.pkl"))
        
        # Préparation et sauvegarde des données pour Sentiment
        sentiment_data = preprocessor.prepare_for_bc_lstm(processed_data, mode='sentiment')
        preprocessor.save_processed_data(sentiment_data, os.path.join(pickles_dir, "MELD_sentiment.pkl"))
        
        print("Prétraitement terminé avec succès!")
    except Exception as e:
        print(f"Erreur lors du prétraitement: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()