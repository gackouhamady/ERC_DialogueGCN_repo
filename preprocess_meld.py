import os
import pickle
import numpy as np
import pandas as pd
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


def create_utterances(filename: str, split: str, max_sent_length: int):
    """
    Lit un fichier CSV MELD et retourne un DataFrame de phrases prétraitées
    avec colonnes : sentence, emotion_label, sentiment_label, speaker, conv_id, utt_id
    """
    df = pd.read_csv(filename)

    records = []
    for _, row in df.iterrows():
        convo_id = f"{split[:2]}_d{int(row['Dialogue_ID'])}"
        utt_id = f"{convo_id}_u{int(row['Utterance_ID'])}"
        text = preprocess_text(str(row['Utterance']))
        # Tokenize later
        records.append({
            'sentence': text,
            'emotion_label': row['Emotion'],  # ex: 'joy'
            'sentiment_label': row['Sentiment'],  # ex: 'positive'
            'speaker': str(row['Speaker']),
            'conv_id': convo_id,
            'utt_id': utt_id
        })

    data = pd.DataFrame(records)
    return data


def load_pretrained_glove(glove_path: str) -> dict:
    """
    Charge les embeddings GloVe depuis un fichier texte.
    """
    glove_path = os.path.abspath(glove_path)
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
            except ValueError:
                continue
    print("GloVe loaded.")
    return embeddings


def encode_labels(labels: pd.Series) -> (dict, dict, np.ndarray):
    """
    Encode les labels texte en entiers et retourne encoders et vecteur étiquettes.
    """
    unique = sorted(labels.unique())
    encoder = {lab: i for i, lab in enumerate(unique)}
    decoder = {i: lab for lab, i in encoder.items()}
    encoded = labels.map(lambda x: encoder[x]).to_numpy()
    return encoder, decoder, encoded


if __name__ == '__main__':
    # Paramètres
    data_dir = 'data/MELD_csv'
    glove_file = 'glove/glove.6B.300d.txt'
    max_num_tokens = 50  # longueur max d'une phrase

    # Chargement et création des DataFrames
    train_df = create_utterances(os.path.join(data_dir, 'train_sent_emo.csv'), 'train', max_num_tokens)
    val_df   = create_utterances(os.path.join(data_dir, 'dev_sent_emo.csv'),   'valid', max_num_tokens)
    test_df  = create_utterances(os.path.join(data_dir, 'test_sent_emo.csv'),  'test',  max_num_tokens)

    # Encodage des labels
    act_enc, act_dec, y_sent_train = encode_labels(train_df['sentiment_label'])
    emo_enc, emo_dec, y_emo_train  = encode_labels(train_df['emotion_label'])
    # Pour valid et test, map existant
    val_df['encoded_sentiment'] = val_df['sentiment_label'].map(act_enc)
    test_df['encoded_sentiment'] = test_df['sentiment_label'].map(act_enc)
    val_df['encoded_emotion']   = val_df['emotion_label'].map(emo_enc)
    test_df['encoded_emotion']  = test_df['emotion_label'].map(emo_enc)
    train_df['encoded_sentiment'] = y_sent_train
    train_df['encoded_emotion']   = y_emo_train

    # Sauvegarde des encodeurs
    os.makedirs('data/MELD/encoders', exist_ok=True)
    pickle.dump(act_enc, open('data/MELD/encoders/sentiment_encoder.pkl', 'wb'))
    pickle.dump(act_dec, open('data/MELD/encoders/sentiment_decoder.pkl', 'wb'))
    pickle.dump(emo_enc, open('data/MELD/encoders/emotion_encoder.pkl', 'wb'))
    pickle.dump(emo_dec, open('data/MELD/encoders/emotion_decoder.pkl', 'wb'))

    # Tokenisation
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['sentence'])
    pickle.dump(tokenizer, open('data/MELD/encoders/tokenizer.pkl', 'wb'))

    # Séquences et padding
    train_seq = tokenizer.texts_to_sequences(train_df['sentence'])
    val_seq   = tokenizer.texts_to_sequences(val_df['sentence'])
    test_seq  = tokenizer.texts_to_sequences(test_df['sentence'])
    train_pad = pad_sequences(train_seq, maxlen=max_num_tokens, padding='post')
    val_pad   = pad_sequences(val_seq,   maxlen=max_num_tokens, padding='post')
    test_pad  = pad_sequences(test_seq,  maxlen=max_num_tokens, padding='post')

    train_df['sequence'] = list(train_pad)
    val_df['sequence']   = list(val_pad)
    test_df['sequence']  = list(test_pad)
    train_df['sent_length'] = train_df['sequence'].apply(len)
    val_df['sent_length']   = val_df['sequence'].apply(len)
    test_df['sent_length']  = test_df['sequence'].apply(len)

    # Préparation par conversation
    conv_ids = set(train_df['conv_id']) | set(val_df['conv_id']) | set(test_df['conv_id'])
    conv_speakers = {}
    conv_sequences = {}
    conv_sentiments = {}
    conv_emotions = {}
    conv_max_len = {}

    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print('Grouping by conversation...')
    for cid in conv_ids:
        sub = all_data[all_data['conv_id'] == cid]
        conv_speakers[cid]   = list(sub['speaker'])
        conv_sequences[cid]  = list(sub['sequence'])
        conv_sentiments[cid] = list(sub['encoded_sentiment'])
        conv_emotions[cid]   = list(sub['encoded_emotion'])
        conv_max_len[cid]    = max(sub['sent_length'])

    # Sauvegarde des données prétraitées
    os.makedirs('data/MELD/processed', exist_ok=True)
    pickle.dump({
        'speakers': conv_speakers,
        'sequences': conv_sequences,
        'max_lengths': conv_max_len,
        'sentiments': conv_sentiments,
        'emotions': conv_emotions,
        'conv_ids': list(conv_ids)
    }, open('data/MELD/processed/MELD_dialogues.pkl', 'wb'))

    # Création de la matrice d'embeddings
    embeddings_index = load_pretrained_glove(glove_file)
    vocab_size = len(tokenizer.word_index) + 1
    emb_dim = len(next(iter(embeddings_index.values())))
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    inv_index = {i: w for w, i in tokenizer.word_index.items()}

    print('Building embedding matrix...')
    for word, idx in tokenizer.word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
        else:
            embedding_matrix[idx] = np.random.randn(emb_dim) * 0.01

    np.save('data/MELD/processed/glove_embedding_matrix.npy', embedding_matrix)
    print('Preprocessing MELD terminé.')
