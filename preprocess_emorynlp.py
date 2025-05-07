#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy  as np
import pandas as pd
from ast import literal_eval
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text: str) -> str:
    """
    Nettoie et normalise le texte : enlève la ponctuation, met en minuscules.
    """
    for punct in '"!&?.,}-/<>#$%\\()*+:;=?@[\\]^_`|\\~':
        text = text.replace(punct, ' ')
    text = ' '.join(text.split())
    return text.lower()

def create_utterances(filename: str, split: str) -> pd.DataFrame:
    """
    Lit un CSV EmoryNLP et retourne un DataFrame avec :
      sentence, emotion_label, speaker (str), conv_id, utt_id
    """
    df = pd.read_csv(filename)
    # Colonnes fixes dans EmoryNLP
    text_col       = 'Utterance'
    speaker_col    = 'Speaker'
    emotion_col    = 'Emotion'
    scene_col      = 'Scene_ID'
    uttid_col      = 'Utterance_ID'
    # Vérif
    for c in (text_col, speaker_col, emotion_col, scene_col, uttid_col):
        if c not in df.columns:
            raise RuntimeError(f"{filename} ne contient pas la colonne '{c}'")

    records = []
    for _, row in df.iterrows():
        # Construction des IDs
        scene = row[scene_col]
        utt   = row[uttid_col]
        conv_id = f"{split[:2]}_sc{scene}"
        utt_id  = f"{conv_id}_u{utt}"

        # Nettoyage du texte
        sent = preprocess_text(str(row[text_col]))

        # Extraction du speaker (c'est une liste sous forme de chaîne)
        try:
            spk_list = literal_eval(row[speaker_col])
            speaker  = spk_list[0] if spk_list else str(row[speaker_col])
        except:
            speaker = str(row[speaker_col])

        records.append({
            'sentence'      : sent,
            'emotion_label' : row[emotion_col],
            'speaker'       : speaker,
            'conv_id'       : conv_id,
            'utt_id'        : utt_id
        })

    return pd.DataFrame(records)

def load_pretrained_glove(glove_path: str) -> dict:
    """
    Charge les embeddings GloVe depuis un .txt.
    """
    print(f"Loading GloVe from {glove_path}…")
    embeddings = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word  = parts[0]
            try:
                vec = np.asarray(parts[1:], dtype='float32')
                embeddings[word] = vec
            except:
                continue
    print("GloVe loaded.")
    return embeddings

def encode_labels(labels: pd.Series):
    """
    Encode des étiquettes texte -> entiers.
    Renvoie (encoder_dict, decoder_dict, numpy_array).
    """
    uniques = sorted(labels.unique())
    enc = {lab: i for i, lab in enumerate(uniques)}
    dec = {i: lab for lab, i in enc.items()}
    arr = labels.map(lambda x: enc[x]).to_numpy()
    return enc, dec, arr

if __name__ == '__main__':
    # ---- Config ----
    data_dir   = 'data/EmoryNLP_csv'
    glove_file = 'glove/glove.6B.300d.txt'
    max_tokens = 50

    # 1) Lecture et prétraitement
    train_df = create_utterances(os.path.join(data_dir, 'emorynlp_train_final.csv'), 'train')
    val_df   = create_utterances(os.path.join(data_dir, 'emorynlp_dev_final.csv'),   'valid')
    test_df  = create_utterances(os.path.join(data_dir, 'emorynlp_test_final.csv'),  'test')

    # 2) Encodage des émotions (train → valid/test)
    emo_enc, emo_dec, y_train = encode_labels(train_df['emotion_label'])
    for df in (val_df, test_df):
        df['encoded_emotion'] = df['emotion_label'].map(emo_enc)
    train_df['encoded_emotion'] = y_train

    # 3) Sauvegarde des encodeurs
    os.makedirs('data/EmoryNLP/encoders', exist_ok=True)
    pickle.dump(emo_enc, open('data/EmoryNLP/encoders/emotion_encoder.pkl','wb'))
    pickle.dump(emo_dec, open('data/EmoryNLP/encoders/emotion_decoder.pkl','wb'))

    # 4) Tokenisation sur les phrases du train
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['sentence'])
    pickle.dump(tokenizer, open('data/EmoryNLP/encoders/tokenizer.pkl','wb'))

    # 5) Séquences + padding
    seq_train = tokenizer.texts_to_sequences(train_df['sentence'])
    seq_val   = tokenizer.texts_to_sequences(val_df['sentence'])
    seq_test  = tokenizer.texts_to_sequences(test_df['sentence'])

    pad_train = pad_sequences(seq_train, maxlen=max_tokens, padding='post')
    pad_val   = pad_sequences(seq_val,   maxlen=max_tokens, padding='post')
    pad_test  = pad_sequences(seq_test,  maxlen=max_tokens, padding='post')

    train_df['sequence']    = list(pad_train)
    val_df['sequence']      = list(pad_val)
    test_df['sequence']     = list(pad_test)
    train_df['sent_length'] = train_df['sequence'].apply(len)
    val_df['sent_length']   = val_df['sequence'].apply(len)
    test_df['sent_length']  = test_df['sequence'].apply(len)

    # 6) Groupage par conversation (Scene_ID)
    conv_ids        = set(train_df['conv_id']) | set(val_df['conv_id']) | set(test_df['conv_id'])
    conv_speakers   = {}
    conv_sequences  = {}
    conv_emotions   = {}
    conv_max_len    = {}

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print("Grouping by conversation…")
    for cid in conv_ids:
        sub = all_df[all_df['conv_id'] == cid]
        conv_speakers[cid]  = list(sub['speaker'])
        conv_sequences[cid] = list(sub['sequence'])
        conv_emotions[cid]  = list(sub['encoded_emotion'])
        conv_max_len[cid]   = max(sub['sent_length'])

    # 7) Sauvegarde du pickle final
    os.makedirs('data/EmoryNLP/processed', exist_ok=True)
    pickle.dump({
        'speakers'    : conv_speakers,
        'sequences'   : conv_sequences,
        'max_lengths' : conv_max_len,
        'emotions'    : conv_emotions,
        'conv_ids'    : list(conv_ids)
    }, open('data/EmoryNLP/processed/EmoryNLP_dialogues.pkl','wb'))
    print("Saved EmoryNLP pickle.")

    # 8) Matrice GloVe
    emb_index = load_pretrained_glove(glove_file)
    vocab_sz  = len(tokenizer.word_index) + 1
    emb_dim   = len(next(iter(emb_index.values())))
    emb_mat   = np.zeros((vocab_sz, emb_dim))
    print("Building embedding matrix…")
    for w,i in tokenizer.word_index.items():
        vec = emb_index.get(w)
        emb_mat[i] = vec if vec is not None else np.random.randn(emb_dim)*0.01

    np.save('data/EmoryNLP/processed/glove_embedding_matrix.npy', emb_mat)
    print("Preprocessing EmoryNLP terminé.")
