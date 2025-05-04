import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.baseline_daily_dialog import DialogueGCN_DailyModel, GRUModel, LSTMModel
import numpy as np, argparse, time, pickle, random
import torch
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f"Running on device: {device}")
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def preprocess_text(text: str) -> str:
    """
    Nettoie et normalise le texte : enlève la ponctuation, met en minuscules.
    """
    for punct in '"!&?.,}-/<>#$%\\()*+:;=?@[\\]^_`|\\~':
        text = text.replace(punct, ' ')
    text = ' '.join(text.split())
    return text.lower()

class MELDDataset(Dataset):
    """
    Dataset PyTorch pour les dialogues MELD (émotion et sentiment) prétraités.
    Charge un pickle contenant :
      - speakers: dict conv_id -> list[str]
      - sequences: dict conv_id -> list[list[int]]
      - sentiments: dict conv_id -> list[int]
      - emotions: dict conv_id -> list[int]
      - max_lengths: dict conv_id -> int
      - conv_ids: list[str]
    Les conv_id sont préfixés par 'tr', 'va', 'te' pour train, valid, test.
    """

    def __init__(self, split: str, path: str):
        data = pickle.load(open(path, 'rb'))
        self.speakers = data['speakers']
        self.sequences = data['sequences']
        self.sentiments = data['sentiments']
        self.emotions = data['emotions']
        self.max_lengths = data['max_lengths']
        self.conv_ids = data['conv_ids']
        if split == 'train': prefix = 'tr'
        elif split == 'valid': prefix = 'va'
        elif split == 'test':  prefix = 'te'
        else: raise ValueError(f"Split inconnu: {split}")
        self.keys = [cid for cid in self.conv_ids if cid.startswith(prefix)]
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        cid = self.keys[index]
        seq = torch.LongTensor(self.sequences[cid])
        spk = torch.FloatTensor([[1, 0] if s == '0' else [0, 1] for s in self.speakers[cid]])
        mask = torch.FloatTensor([1] * len(self.emotions[cid]))
        emotion = torch.LongTensor(self.emotions[cid])
        sentiment = torch.LongTensor(self.sentiments[cid])
        return seq, spk, mask, emotion, sentiment, cid

    def collate_fn(self, batch):
        seqs, spks, masks, emos, sents, cids = zip(*batch)
        seq_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
        spk_pad = pad_sequence(spks, batch_first=True, padding_value=0)
        mask_pad = pad_sequence([m.unsqueeze(1) for m in masks], batch_first=True, padding_value=0).squeeze(-1)
        emo_pad = pad_sequence(emos, batch_first=True, padding_value=-100)
        sent_pad = pad_sequence(sents, batch_first=True, padding_value=-100)
        return seq_pad, spk_pad, mask_pad, emo_pad, sent_pad, list(cids)

class MELDDataset2(Dataset):
    """Variante retournant aussi max_seq_len."""
    def __init__(self, split: str, path: str):
        data = pickle.load(open(path, 'rb'))
        self.speakers = data['speakers']; self.sequences = data['sequences']
        self.sentiments = data['sentiments']; self.emotions = data['emotions']
        self.max_lengths = data['max_lengths']; self.conv_ids = data['conv_ids']
        if split == 'train': prefix = 'tr'
        elif split == 'valid': prefix = 'va'
        elif split == 'test':  prefix = 'te'
        else: raise ValueError(f"Split inconnu: {split}")
        self.keys = [cid for cid in self.conv_ids if cid.startswith(prefix)]; self.len = len(self.keys)
    def __len__(self): return self.len
    def __getitem__(self, index):
        cid = self.keys[index]
        seq = torch.LongTensor(self.sequences[cid])
        spk = torch.FloatTensor([[1, 0] if s == '0' else [0, 1] for s in self.speakers[cid]])
        mask = torch.FloatTensor([1] * len(self.emotions[cid]))
        emotion = torch.LongTensor(self.emotions[cid])
        sentiment = torch.LongTensor(self.sentiments[cid])
        return seq, spk, mask, emotion, sentiment, self.max_lengths[cid], cid
    def collate_fn(self, batch):
        seqs, spks, masks, emos, sents, maxlens, cids = zip(*batch)
        seq_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
        spk_pad = pad_sequence(spks, batch_first=True, padding_value=0)
        mask_pad = pad_sequence([m.unsqueeze(1) for m in masks], batch_first=True, padding_value=0).squeeze(-1)
        emo_pad = pad_sequence(emos, batch_first=True, padding_value=-100)
        sent_pad = pad_sequence(sents, batch_first=True, padding_value=-100)
        return seq_pad, spk_pad, mask_pad, emo_pad, sent_pad, list(maxlens), list(cids)


def get_meld_loaders(path: str, batch_size: int=32, num_workers: int=0, pin_memory: bool=False):
    trainset = MELDDataset2('train', path); validset = MELDDataset2('valid', path); testset = MELDDataset2('test', path)
    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, collate_fn=validset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(testset,  batch_size=batch_size, collate_fn=testset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

# Device setup
# Après parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f"Running on device: {device}")

def process_meld_data_loader(batch):
    textf, spk_mask, umask, emo_labels, sent_labels, max_seq_lens, cids = batch
    L = max(max_seq_lens)
    input_sequence = textf[:, :, :L]
    return [input_sequence.to(device), spk_mask.to(device), umask.to(device), emo_labels.to(device)]


def train_or_eval_graph_model(
    model,
    loss_function,
    dataloader,
    epoch: int,
    cuda: bool,
    optimizer=None,
    train: bool=False
):
    """
    Entraîne ou évalue le modèle GCN sur MELD.
    """
    device_str = 'cuda' if cuda else 'cpu'
    model.train() if train else model.eval()

    all_losses, all_preds, all_labels, all_vids = [], [], [], []
    ei = torch.empty(0, device=device_str)
    et = torch.empty(0, device=device_str)
    en = torch.empty(0, device=device_str)
    el = []

    for step, data in enumerate(dataloader):
        if train: optimizer.zero_grad()
        # Unpack and preprocess batch
        textf_raw, qmask_raw, umask_raw, emo_raw, sent_raw, maxlens_raw, vid_raw = data
        textf, qmask, umask, labels = process_meld_data_loader(data)
        B, T = umask.size()
        lengths = [int(umask[b].sum().item()) for b in range(B)]

        # Forward
        log_prob, e_i_b, e_n_b, e_t_b, e_l_b = model(textf, qmask, umask, lengths)
        N_pred, C = log_prob.shape

        # Flatten labels & mask
        flat_labels = labels.contiguous().view(-1)
        flat_umask = umask.contiguous().view(-1)
        kept_pos = int(flat_umask.sum().item())

        # Align preds vs labels
        while N_pred != kept_pos:
            diff = kept_pos - N_pred
            if diff > 0:
                last = log_prob[-1:].detach()
                pad  = last.expand(diff, C)
                log_prob = torch.cat([log_prob, pad], 0)
            else:
                cut = -diff
                idxs = (flat_umask==1).nonzero(as_tuple=False).squeeze(1)
                drop = idxs[-cut:]
                mask_keep = torch.ones_like(flat_umask, dtype=torch.bool)
                mask_keep[drop] = False
                flat_labels = flat_labels[mask_keep]
                flat_umask  = flat_umask[mask_keep]
            N_pred = log_prob.size(0)
            kept_pos = int(flat_umask.sum().item())

        label = flat_labels[flat_umask==1]
        assert label.numel()==N_pred, "Alignement final échoué"

        # Loss & backward
        loss = loss_function(log_prob, label.to(device_str))
        all_losses.append(loss.item())
        ei = torch.cat([ei, e_i_b], dim=1)
        et = torch.cat([et, e_t_b], dim=0)
        en = torch.cat([en, e_n_b], dim=0)
        el += e_l_b
        all_preds.append(log_prob.argmax(dim=1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_vids.extend(vid_raw)

        if train:
            loss.backward(); optimizer.step()
            if step%10==0:
                ps = np.concatenate(all_preds); ls = np.concatenate(all_labels)
                print(f"Step {step} | loss={np.mean(all_losses):.4f} | acc={accuracy_score(ls,ps)*100:.2f}% | f1={f1_score(ls,ps,average='micro')*100:.2f}%")

    # Metrics aggregation
    if not all_preds:
        return [float('nan')]*12
    preds = np.concatenate(all_preds); labels = np.concatenate(all_labels)
    avg_loss = float(np.mean(all_losses))
    avg_acc  = accuracy_score(labels,preds)*100
    avg_f1   = f1_score(labels,preds,average='micro')*100
    avg_prec = precision_score(labels,preds,average='micro')*100
    avg_rec  = recall_score(labels,preds,average='micro')*100

    return (avg_loss, avg_acc, labels, preds, avg_f1, np.array(all_vids),
            ei.cpu().numpy(), et.cpu().numpy(), en.detach().cpu().numpy(), np.array(el), avg_prec, avg_rec)




if __name__ == '__main__':
    import argparse
    import datetime as dt
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import numpy as np
    import time
    from models.baseline_meld import (
        DialogueGCN_MELDModel,
        GRUModel,
        LSTMModel,
        MaskedNLLLoss,
    )

    # --- Arguments CLI ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda',         action='store_true', help='ne pas utiliser le GPU')
    parser.add_argument('--base-model',      default='LSTM',      help='DialogRNN/LSTM/GRU')
    parser.add_argument('--graph-model',     action='store_true', default=True,  help='appliquer GCN après encodage')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='utiliser nodal attention')
    parser.add_argument('--windowp',         type=int, default=10, help='taille fenêtre passée')
    parser.add_argument('--windowf',         type=int, default=10, help='taille fenêtre future')
    parser.add_argument('--lr',              type=float, default=1e-4,   help='learning rate')
    parser.add_argument('--l2',              type=float, default=1e-5,   help='L2 weight decay')
    parser.add_argument('--dropout',         type=float, default=0.5,    help='dropout rate')
    parser.add_argument('--batch-size',      type=int,   default=32,     help='taille de batch')
    parser.add_argument('--epochs',          type=int,   default=5,      help='nombre d\'épochs')
    parser.add_argument('--tensorboard',     action='store_true', default=False, help='log TensorBoard')
    parser.add_argument('--save-path',       type=str, default=f'./saved/DialogueGCN_MELD_{dt.datetime.now():%m%d_%H%M}', help='répertoire de sauvegarde')
    args = parser.parse_args()

    # Device
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f"Running on device: {device}")

    # Paramètres généraux
    n_classes     = 7  # MELD a 7 émotions, mais on ne le passe plus au constructeur
    D_m, D_g      = 100, 150
    D_p, D_e      = 150, 100
    D_h, D_a      = 100, 100
    graph_hidden_size = 100  # correspond à votre ancien graph_h

    # Chargement des embeddings (GloVe)
    glv_matrix = np.load('../data/MELD/processed/glove_embedding_matrix.npy')  # (vocab_size, embedding_dim)
    vocab_size, embedding_dim = glv_matrix.shape

    if args.graph_model:
        seed_everything()
        model = DialogueGCN_MELDModel(
            base_model        = args.base_model,
            D_m               = D_m,
            D_g               = D_g,
            D_p               = D_p,
            D_e               = D_e,
            D_h               = D_h,
            D_a               = D_a,
            graph_hidden_size = graph_hidden_size,    # remplacé ici
            n_speakers        = 2,
            max_seq_len       = 50,
            window_past       = args.windowp,
            window_future     = args.windowf,
            vocab_size        = vocab_size,
            embedding_dim     = embedding_dim,
            dropout           = args.dropout,
            nodal_attention   = args.nodal_attention,
            no_cuda           = args.no_cuda
        )
        model.init_pretrained_embeddings(glv_matrix)
        print('DialogueGCN sur MELD avec', args.base_model)
    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
        else:
            raise NotImplementedError("base model must be GRU or LSTM")
        print(f'Basic {args.base_model} Model sur MELD')

    if args.cuda:
        model.to(device)


    # --- CONFIGURATION ENTRAÎNEMENT MELD ---

    # 1) Fonction de perte et optimiseur
    loss_function = nn.NLLLoss() if args.graph_model else MaskedNLLLoss()
    optimizer     = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # 2) DataLoaders MELD
    data_path = '../data/MELD/processed/MELD_dialogues.pkl'
    train_loader, valid_loader, test_loader = get_meld_loaders(
        data_path,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=args.cuda
    )
    print(f"[DEBUG] batches ▶ train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")

    # 3) Boucle d'entraînement / évaluation
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # — Train —
        tr_loss, tr_acc, _, _, tr_f1, _, *_ = train_or_eval_graph_model(
            model, loss_function, train_loader, epoch, args.cuda, optimizer, train=True
        ) if args.graph_model else train_or_eval_graph_model(
            model, loss_function, train_loader, epoch, optimizer, train=True
        )

        # — Validation —
        va_loss, va_acc, _, _, va_f1, _, *_ = train_or_eval_graph_model(
            model, loss_function, valid_loader, epoch, args.cuda, optimizer=None, train=False
        ) if args.graph_model else  train_or_eval_graph_model(
            model, loss_function, valid_loader, epoch
        )

        # — Test —
        te_loss, te_acc, te_labels, te_preds, te_f1, *_ = train_or_eval_graph_model(
            model, loss_function, test_loader, epoch, args.cuda, optimizer=None, train=False
        ) if args.graph_model else  train_or_eval_graph_model(
            model, loss_function, test_loader, epoch
        )

        # — Logs —
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"tr_loss={tr_loss:.4f}, tr_acc={tr_acc:.2f}%, tr_f1={tr_f1:.2f}% | "
              f"va_loss={va_loss:.4f}, va_acc={va_acc:.2f}%, va_f1={va_f1:.2f}% | "
              f"te_loss={te_loss:.4f}, te_acc={te_acc:.2f}%, te_f1={te_f1:.2f}% | "
              f"time={(time.time() - start):.1f}s")

        # — Sauvegarde du meilleur modèle —
        if te_f1 > best_f1:
            best_f1 = te_f1
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))
            print(f"→ Nouveau meilleur F1: {best_f1:.2f}%, modèle sauvegardé.")

    print("Entraînement terminé.")
    # Pour réutilisation ultérieure, on peut aussi :
    torch.save(model.state_dict(), os.path.join(args.save_path, 'final_model.pt'))
    print("Modèle final enregistré.")



    