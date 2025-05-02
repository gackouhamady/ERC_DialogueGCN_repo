import json
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle
import torch # type: ignore
from transformers import BertModel, BertTokenizer

def preprocess_text(x):
    for punct in '"!&?.,}-/<>#$%\\()*+:;=?@[\\]^_`|~':
        x = x.replace(punct, ' ')
    x = ' '.join(x.split())
    return x.lower()

def create_utterances(filename, split):
    sentences, act_labels, emotion_labels, speakers, conv_id, utt_id = [], [], [], [], [], []
    with open(filename, 'r') as f:
        for c_id, line in enumerate(f):
            s = eval(line)
            for u_id, item in enumerate(s['dialogue']):
                sentences.append(item['text'])
                act_labels.append(item['act'])
                emotion_labels.append(item['emotion'])
                conv_id.append(split[:2] + '_c' + str(c_id))
                utt_id.append(split[:2] + '_c' + str(c_id) + '_u' + str(u_id))
                speakers.append(str(u_id % 2))
    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(preprocess_text)
    data['act_label'] = act_labels
    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id
    return data

def encode_labels(encoder, l):
    return encoder[l]

def pad_data(texts, max_num_tokens=250):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_tensor_list, segments_tensor_list, input_masks_tensor_list = [], [], []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        segments = [0] * len(tokens)
        input_mask = [1] * len(tokens)
        # Padding
        if len(tokens) < max_num_tokens:
            pad_len = max_num_tokens - len(tokens)
            tokens += [0] * pad_len
            segments += [0] * pad_len
            input_mask += [0] * pad_len
        else:
            tokens = tokens[:max_num_tokens]
            segments = segments[:max_num_tokens]
            input_mask = input_mask[:max_num_tokens]
        tokens_tensor_list.append(torch.tensor([tokens]))
        segments_tensor_list.append(torch.tensor([segments]))
        input_masks_tensor_list.append(torch.tensor([input_mask]))
    return tokens_tensor_list, segments_tensor_list, input_masks_tensor_list

if __name__ == '__main__':
    # ✅ chemins corrects
    train_data = create_utterances('../daily_dialog_loader/dailydialog_json/dev_1.json', 'train')
    valid_data = create_utterances('../daily_dialog_loader/dailydialog_json/test_1.json', 'valid')
    test_data = create_utterances('../daily_dialog_loader/dailydialog_json/train_1.json', 'test')

    # ✅ encodage des labels
    act_label_encoder = {label: i for i, label in enumerate(set(train_data['act_label']))}
    emotion_label_encoder = {label: i for i, label in enumerate(set(train_data['emotion_label']))}
    print("Act labels:", act_label_encoder)
    print("Emotion labels:", emotion_label_encoder)

    for df in [train_data, valid_data, test_data]:
        df['encoded_act_label'] = df['act_label'].map(lambda x: act_label_encoder[x])
        df['encoded_emotion_label'] = df['emotion_label'].map(lambda x: emotion_label_encoder[x])

    # ✅ tokenization BERT
    print("Tokenization et padding...")
    train_tokens, train_segments, train_masks = pad_data(list(train_data['sentence']))
    valid_tokens, valid_segments, valid_masks = pad_data(list(valid_data['sentence']))
    test_tokens, test_segments, test_masks = pad_data(list(test_data['sentence']))

    print("Chargement de BERT...")
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    USE_CUDA = torch.cuda.is_available() and torch.version.cuda is not None
    if USE_CUDA:
        model.to('cuda')
        print("🟢 CUDA activé.")
    else:
        print("⚪ Mode CPU.")

    def get_bert_embeddings(tokens_list, segments_list, masks_list):
        results = []
        with torch.no_grad():
            for i in range(len(tokens_list)):
                tokens = tokens_list[i].to('cuda') if USE_CUDA else tokens_list[i]
                segs = segments_list[i].to('cuda') if USE_CUDA else segments_list[i]
                masks = masks_list[i].to('cuda') if USE_CUDA else masks_list[i]
                output = model(tokens, token_type_ids=segs, attention_mask=masks)
                cls_vec = output.last_hidden_state[:, 0, :]  # [CLS] token
                results.append(cls_vec.cpu().numpy()[0])
                if i % 1000 == 0:
                    print(f"Processed {i}")
        return results

    print("Génération des vecteurs BERT...")
    train_seq = get_bert_embeddings(train_tokens, train_segments, train_masks)
    valid_seq = get_bert_embeddings(valid_tokens, valid_segments, valid_masks)
    test_seq  = get_bert_embeddings(test_tokens,  test_segments,  test_masks)

    # ✅ assigner les vecteurs aux DataFrames
    train_data['sequence'] = train_seq
    valid_data['sequence'] = valid_seq
    test_data['sequence']  = test_seq

    # ✅ regroupement par conversation
    convSpeakers, convInputSequence, convActLabels, convEmotionLabels = {}, {}, {}, {}
    all_data = pd.concat([train_data, valid_data, test_data])
    train_ids, valid_ids, test_ids = set(train_data['conv_id']), set(valid_data['conv_id']), set(test_data['conv_id'])

    for conv in train_ids | valid_ids | test_ids:
        df = all_data[all_data['conv_id'] == conv]
        convSpeakers[conv] = list(df['speaker'])
        convInputSequence[conv] = list(df['sequence'])
        convActLabels[conv] = list(df['encoded_act_label'])
        convEmotionLabels[conv] = list(df['encoded_emotion_label'])

    # ✅ sauvegarde
    print("Sauvegarde du fichier pickle...")
    os.makedirs('dailydialog', exist_ok=True)
    with open('dailydialog/daily_dialogue_bert2.pkl', 'wb') as f:
        pickle.dump([
            convSpeakers, convInputSequence, convActLabels, convEmotionLabels,
            train_ids, test_ids, valid_ids
        ], f)

    print("✅ Prétraitement terminé. Fichier enregistré dans dailydialog/daily_dialogue_bert2.pkl")
