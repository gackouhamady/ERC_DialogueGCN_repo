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



class MORIDataset(Dataset):
    """
    Dataset PyTorch pour MORI NLP (émotion) prétraité en pickle.
    Le pickle doit contenir :
      - speakers: dict conv_id -> list[str]
      - sequences: dict conv_id -> list[list[int]]
      - emotions: dict conv_id -> list[int]
      - max_lengths: dict conv_id -> int
      - conv_ids: list[str]
    Les conv_id sont préfixés par 'tr', 'va', 'te'.
    """
    def __init__(self, split: str, pkl_path: str):
        data = pickle.load(open(pkl_path, 'rb'))
        self.speakers    = data['speakers']
        self.sequences   = data['sequences']
        self.emotions    = data['emotions']
        self.max_lengths = data['max_lengths']
        self.conv_ids    = data['conv_ids']
        if split == 'train': prefix = 'tr'
        elif split == 'valid': prefix = 'va'
        elif split == 'test':  prefix = 'te'
        else: raise ValueError(f"Split inconnu: {split}")
        self.keys = [cid for cid in self.conv_ids if cid.startswith(prefix)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        cid   = self.keys[idx]
        seq   = torch.LongTensor(self.sequences[cid])
        spk   = torch.FloatTensor([[1,0] if s=='0' else [0,1] for s in self.speakers[cid]])
        mask  = torch.FloatTensor([1] * len(self.emotions[cid]))
        emo   = torch.LongTensor(self.emotions[cid])
        return seq, spk, mask, emo, cid

    def collate_fn(self, batch):
        seqs, spks, masks, emos, cids = zip(*batch)
        seq_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
        spk_pad = pad_sequence(spks, batch_first=True, padding_value=0)
        mask_pad = pad_sequence([m.unsqueeze(1) for m in masks], batch_first=True, padding_value=0).squeeze(-1)
        emo_pad  = pad_sequence(emos, batch_first=True, padding_value=-100)
        return seq_pad, spk_pad, mask_pad, emo_pad, list(cids)
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MORIDataset2(Dataset):
    """
    Variante de MORIDataset retournant aussi max_seq_len par dialogue.
    Tronque chaque dialogue à max_utts énoncés, et convertit les noms de speakers
    en vecteurs one-hot via spk2idx.
    """
    def __init__(
        self,
        split: str,
        pkl_path: str,
        spk2idx: dict,
        max_utts: int = 100
    ):
        data = pickle.load(open(pkl_path, 'rb'))
        self.speakers    = data['speakers']    # dict conv_id -> list[str]
        self.sequences   = data['sequences']   # dict conv_id -> list[list[int]]
        self.emotions    = data['emotions']    # dict conv_id -> list[int]
        self.max_lengths = data['max_lengths'] # dict conv_id -> int
        self.conv_ids    = data['conv_ids']
        self.spk2idx     = spk2idx
        self.max_utts    = max_utts

        if split == 'train': prefix = 'tr'
        elif split == 'valid': prefix = 'va'
        elif split == 'test':  prefix = 'te'
        else:
            raise ValueError(f"Split inconnu: {split}")

        self.keys = [cid for cid in self.conv_ids if cid.startswith(prefix)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        cid = self.keys[idx]
        seqs = self.sequences[cid]
        spks = self.speakers[cid]
        emos = self.emotions[cid]
        maxlen = self.max_lengths[cid]

        # Tronquage si dialogue trop long
        if len(seqs) > self.max_utts:
            seqs  = seqs[-self.max_utts:]
            spks  = spks[-self.max_utts:]
            emos  = emos[-self.max_utts:]

        # 1) séquences de tokens → LongTensor (U, L)
        seq = torch.LongTensor(seqs)

        # 2) locuteurs → one-hot FloatTensor (U, n_speakers)
        #    on suppose spk2idx couvre tous les noms rencontrés
        idxs = [self.spk2idx[name] for name in spks]
        n_spk = len(self.spk2idx)
        spk  = torch.zeros(len(idxs), n_spk, dtype=torch.float)
        for i, spk_idx in enumerate(idxs):
            spk[i, spk_idx] = 1.0

        # 3) masque et labels
        mask    = torch.ones(len(emos), dtype=torch.float)
        emotion = torch.LongTensor(emos)

        return seq, spk, mask, emotion, maxlen, cid

    def collate_fn(self, batch):
        seqs, spks, masks, emos, maxlens, cids = zip(*batch)
        seq_pad  = pad_sequence(seqs,  batch_first=True, padding_value=0)    # (B, U_max, L)
        spk_pad  = pad_sequence(spks,  batch_first=True, padding_value=0)    # (B, U_max, n_spk)
        mask_pad = pad_sequence(
                       [m.unsqueeze(1) for m in masks],
                       batch_first=True, padding_value=0
                   ).squeeze(-1)                                            # (B, U_max)
        emo_pad  = pad_sequence(emos,  batch_first=True, padding_value=-100) # (B, U_max)
        # return seq_pad, spk_pad, mask_pad, emo_pad, list(maxlens), list(cids)
        return seq_pad, spk_pad, mask_pad, emo_pad, maxlens, spk_pad, cids
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class MORIDataset2(Dataset):
    """
    Variante de MORIDataset retournant :
      - seq_pad      : LongTensor [U, L]
      - spk_pad      : FloatTensor [U, n_speakers] (one-hot)
      - mask         : FloatTensor [U]
      - emotion      : LongTensor [U]
      - maxlen       : int  (longueur originale avant padding)
      - conv_id      : str
    Tronque chaque dialogue à max_utts énoncés.
    """
    def __init__(
        self,
        split: str,
        pkl_path: str,
        spk2idx: dict,
        max_utts: int = 100
    ):
        data = pickle.load(open(pkl_path, 'rb'))
        self.speakers    = data['speakers']    # dict conv_id -> list[str]
        self.sequences   = data['sequences']   # dict conv_id -> list[list[int]]
        self.emotions    = data['emotions']    # dict conv_id -> list[int]
        self.max_lengths = data['max_lengths'] # dict conv_id -> int
        self.conv_ids    = data['conv_ids']    # list[str]
        self.spk2idx     = spk2idx             # mapping speaker_name -> index
        self.max_utts    = max_utts

        if split == 'train':   prefix = 'tr'
        elif split == 'valid': prefix = 'va'
        elif split == 'test':  prefix = 'te'
        else:
            raise ValueError(f"Split inconnu: {split}")

        self.keys = [cid for cid in self.conv_ids if cid.startswith(prefix)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        cid    = self.keys[idx]
        seqs   = self.sequences[cid]
        spks   = self.speakers[cid]
        emos   = self.emotions[cid]
        maxlen = self.max_lengths[cid]

        # Tronquage si trop long
        if len(seqs) > self.max_utts:
            seqs   = seqs[-self.max_utts:]
            spks   = spks[-self.max_utts:]
            emos   = emos[-self.max_utts:]

        # 1) séquences de tokens → LongTensor (U, L)
        seq = torch.LongTensor(seqs)

        # 2) locuteurs → one-hot FloatTensor (U, n_speakers)
        idxs = [ self.spk2idx[name] for name in spks ]
        n_spk = len(self.spk2idx)
        spk   = torch.zeros(len(idxs), n_spk, dtype=torch.float)
        for i, sidx in enumerate(idxs):
            spk[i, sidx] = 1.0

        # 3) masque et labels
        mask    = torch.ones(len(emos), dtype=torch.float)  # (U,)
        emotion = torch.LongTensor(emos)                    # (U,)

        return seq, spk, mask, emotion, maxlen, cid

    def collate_fn(self, batch):
        # unpack 6-uplets depuis __getitem__
        seqs, spks, masks, emos, maxlens, cids = zip(*batch)

        # pad sequences
        seq_pad  = pad_sequence(seqs,  batch_first=True, padding_value=0)    # (B, U_max, L)
        spk_pad  = pad_sequence(spks,  batch_first=True, padding_value=0)    # (B, U_max, n_speakers)
        mask_pad = pad_sequence(
                       [m.unsqueeze(1) for m in masks],
                       batch_first=True, padding_value=0
                   ).squeeze(-1)                                            # (B, U_max)
        emo_pad  = pad_sequence(emos,  batch_first=True, padding_value=-100) # (B, U_max)

        # => on renvoie maintenant 7 éléments, en y incluant spk_pad comme "batch_speakers"
        #return seq_pad, spk_pad, mask_pad, emo_pad, list(maxlens), spk_pad, list(cids)
        return seq_pad, spk_pad, mask_pad, emo_pad, list(maxlens), list(cids)


def get_mori_loaders(
    pkl_path: str,
    spk2idx: dict,
    batch_size: int   = 1,
    num_workers: int  = 0,
    pin_memory: bool  = False,
    max_utts: int     = 100
):
    """
    Retourne trois DataLoaders (train, valid, test) pour le MORI NLP Dataset.
    """
    trainset = MORIDataset2('train', pkl_path, spk2idx, max_utts=max_utts)
    validset = MORIDataset2('valid', pkl_path, spk2idx, max_utts=max_utts)
    testset  = MORIDataset2('test',  pkl_path, spk2idx, max_utts=max_utts)

    train_loader = DataLoader(
        trainset,
        batch_size   = batch_size,
        shuffle      = True,
        collate_fn   = trainset.collate_fn,
        num_workers  = num_workers,
        pin_memory   = pin_memory
    )
    valid_loader = DataLoader(
        validset,
        batch_size   = batch_size,
        shuffle      = False,
        collate_fn   = validset.collate_fn,
        num_workers  = num_workers,
        pin_memory   = pin_memory
    )
    test_loader  = DataLoader(
        testset,
        batch_size   = batch_size,
        shuffle      = False,
        collate_fn   = testset.collate_fn,
        num_workers  = num_workers,
        pin_memory   = pin_memory
    )

    return train_loader, valid_loader, test_loader


# Device setup (à placer après parser.parse_args())
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f"Running on device: {device}")

def process_mori_data_loader(batch):
    """
    Prépare un batch MORI pour l’entraînement/évaluation.
    batch: (seq_pad, spk_pad, mask_pad, emo_pad, max_seq_lens, cids)
      - seq_pad: LongTensor [B, U, Lmax]
      - spk_pad: FloatTensor [B, U, 2]
      - mask_pad: FloatTensor [B, U]
      - emo_pad:  LongTensor [B, U]
      - max_seq_lens: list[int] longueur réelle par dialogue
      - cids: list[str]
    """
    seq_pad, spk_pad, mask_pad, emo_pad, max_seq_lens, cids = batch
    
    # on tronque au plus long vrai énoncé de la batch
    L = max(max_seq_lens)
    input_sequence = seq_pad[:, :, :L]        # (B, U, L)
    return [
        input_sequence.to(device),            # [B, U, L]
        spk_pad.to(device),                   # [B, U, 2]
        mask_pad.to(device),                  # [B, U]
        emo_pad.to(device)                    # [B, U]
    ], cids

def train_or_eval_mori_graph_model(
    model,
    loss_function,
    dataloader,
    epoch: int,
    cuda: bool,
    optimizer=None,
    train: bool=False
):
    """
    Entraîne ou évalue le modèle GCN sur MORI.
    Renvoie:
      avg_loss, avg_acc, labels, preds, avg_f1,
      all_conv_ids,
      edge_index_all, edge_type_all, edge_norm_all, edge_len_all,
      avg_prec, avg_rec
    """
    device_str = 'cuda' if cuda else 'cpu'
    model.train() if train else model.eval()

    all_losses, all_preds, all_labels, all_vids = [], [], [], []
    ei_all = torch.empty((0,), device=device_str)
    et_all = torch.empty((0,), device=device_str)
    en_all = torch.empty((0,), device=device_str)
    el_all = []

    for step, batch in enumerate(dataloader, 1):
        if train:
            optimizer.zero_grad()

        # --- DÉBALLAGE CORRECT de vos 6 sorties de collate_fn ---
        # textf, spk_mask, umask, emo_labels, maxlens, cids = batch
        textf, spk_mask, umask, emo_labels, batch_speakers, cids = batch
        B, U = umask.size()
        lengths = [int(umask[b].sum().item()) for b in range(B)]

        # --- PASSAGE EN DEVICE ---
        textf      = textf.to(device_str)      # (B, U, L)
        spk_mask   = spk_mask.to(device_str)   # (B, U, n_speakers)
        umask      = umask.to(device_str)      # (B, U)
        emo_labels = emo_labels.to(device_str) # (B, U)
       # batch_speakers = batch_speakers.to(device_str) # (B, U, n_speakers)
        # maxlens    = maxlens.to(device_str)    # (B,)

        # --- FORWARD (ajustez la signature si vous avez modifié votre forward) ---
        # ici on ne passe PAS maxlens ni cids au modèle,
        # mais si votre forward attend un argument speakers, adaptez.
        # log_prob, e_i_b, e_n_b, e_t_b, e_l_b = model(
          #  textf, spk_mask, umask, lengths
        # )

        log_prob, e_i_b, e_n_b, e_t_b, e_l_b = model(
          textf,spk_mask, umask, lengths, batch_speakers
         )

        # --- APLATISSEMENT pour le calcul de la perte ---
        N_pred, C = log_prob.shape
        flat_labels = emo_labels.contiguous().view(-1)
        flat_umask  = umask.contiguous().view(-1)
        kept_pos    = int(flat_umask.sum().item())

        # réalignement si nécessaire
        while N_pred != kept_pos:
            diff = kept_pos - N_pred
            if diff > 0:
                last = log_prob[-1:].detach()
                pad  = last.expand(diff, C)
                log_prob = torch.cat([log_prob, pad], dim=0)
            else:
                cut  = -diff
                idxs = (flat_umask==1).nonzero(as_tuple=False).squeeze(1)
                drop = idxs[-cut:]
                keep = torch.ones_like(flat_umask, dtype=torch.bool)
                keep[drop] = False
                flat_labels = flat_labels[keep]
                flat_umask  = flat_umask[keep]
            N_pred   = log_prob.size(0)
            kept_pos = int(flat_umask.sum().item())

        label = flat_labels[flat_umask==1]
        assert label.numel() == N_pred, "Alignement final échoué"

        # --- PERTE & BACKWARD ---
        loss = loss_function(log_prob, label.to(device_str))
        all_losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

        # --- AGGLOMÉRATION DES GRAPHS pour analyse ultérieure ---
        ei_all = torch.cat([ei_all, e_i_b.to(device_str)], dim=1)
        et_all = torch.cat([et_all, e_t_b.to(device_str)], dim=0)
        en_all = torch.cat([en_all, e_n_b.to(device_str)], dim=0)
        el_all += e_l_b

        # --- PRÉDICTIONS & MÉTRIQUES BATCH —
        preds_np  = log_prob.argmax(dim=1).cpu().numpy()
        labels_np = label.cpu().numpy()
        all_preds.append(preds_np)
        all_labels.append(labels_np)
        all_vids.extend(cids)

        if train:
            from sklearn.metrics import accuracy_score, f1_score
            batch_acc = accuracy_score(labels_np, preds_np)*100
            batch_f1  = f1_score(labels_np, preds_np, average='micro')*100
            print(f"Step {step}/{len(dataloader)} | "
                  f"loss={loss.item():.4f} | acc={batch_acc:.2f}% | f1={batch_f1:.2f}%")

    # --- AGGLOMÉRATION PAR ÉPOCH — 
    if not all_preds:
        return [float('nan')]*12

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    preds = np.concatenate(all_preds)
    labels= np.concatenate(all_labels)
    avg_loss = float(np.mean(all_losses))
    avg_acc  = accuracy_score(labels, preds)*100
    avg_f1   = f1_score(labels, preds, average='micro')*100
    avg_prec = precision_score(labels, preds, average='micro')*100
    avg_rec  = recall_score(labels, preds, average='micro')*100

    return (
        avg_loss,
        avg_acc,
        labels,
        preds,
        avg_f1,
        np.array(all_vids),
        ei_all.detach().cpu().numpy(),
        et_all.detach().cpu().numpy(),
        en_all.detach().cpu().numpy(),
        np.array(el_all),
        avg_prec,
        avg_rec
    )



if __name__ == '__main__':
    import argparse
    import datetime as dt
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.baseline_emory import (
        DialogueGCN_MORIModel,
        GRUModel,
        LSTMModel,
        MaskedNLLLoss,
    )
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
  
  

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda',         action='store_true', help='ne pas utiliser le GPU')
    parser.add_argument('--base-model',      default='LSTM',      help='LSTM/GRU/None')
    parser.add_argument('--graph-model',     action='store_true', default=True,  help='appliquer GCN après encodage')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='utiliser nodal attention')
    parser.add_argument('--windowp',         type=int, default=1, help='taille fenêtre passée')
    parser.add_argument('--windowf',         type=int, default=1, help='taille fenêtre future')
    parser.add_argument('--lr',              type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2',              type=float, default=1e-5, help='L2 weight decay')
    parser.add_argument('--dropout',         type=float, default=0.5, help='dropout rate')
    parser.add_argument('--batch-size',      type=int,   default=8, help='taille de batch')
    parser.add_argument('--epochs',          type=int,   default=1,  help='nombre d\'épochs')
    parser.add_argument('--save-path',       type=str, default=f'./saved/DialogueGCN_MORI_{dt.datetime.now():%m%d_%H%M}', help='répertoire de sauvegarde')
    args = parser.parse_args()

    # Device setup
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f"Running on device: {device}")

    # Seed
    seed = 100
    def seed_everything(seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_everything()

    # Load emotion encoder
    enc_path = '../data/EmoryNLP/encoders/emotion_encoder.pkl'
    with open(enc_path, 'rb') as f:
        emo_enc = pickle.load(f)
    n_classes = len(emo_enc)
    print(f"Nombre de classes émotionnelles : {n_classes}")

    # Load GloVe embeddings
    glv_matrix = np.load('../data/EmoryNLP/processed/glove_embedding_matrix.npy')
    vocab_size, embedding_dim = glv_matrix.shape
    assert vocab_size == glv_matrix.shape[0], "Incohérence vocab_size et GloVe"

    # Load speakers
    pkl_path = '../data/EmoryNLP/processed/EmoryNLP_dialogues.pkl'
    data = pickle.load(open(pkl_path, 'rb'))
    all_speakers = sorted({ name for spks in data['speakers'].values() for name in spks })
    spk2idx = { name: i for i, name in enumerate(all_speakers) }

    # Load DataLoaders
    train_loader, valid_loader, test_loader = get_mori_loaders(
        pkl_path,  
        spk2idx     = spk2idx, 
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=args.cuda
    )
    print(f"[DEBUG] batches ▶ train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")

    # Init model
    D_m, D_p, D_e, D_h, D_a = 50, 50, 50, 50, 50
    graph_hidden_size = 20

    if args.graph_model:
        model = DialogueGCN_MORIModel(
            base_model        = args.base_model,
            D_m               = D_m,
            D_p               = D_p,
            D_e               = D_e,
            D_h               = D_h,
            D_a               = D_a,
            graph_hidden_size = graph_hidden_size,
            n_classes         = n_classes,
            n_speakers        = len(spk2idx),
            max_seq_len       = 100,
            window_past       = args.windowp,
            window_future     = args.windowf,
            vocab_size        = vocab_size,
            embedding_dim     = embedding_dim,
            dropout           = args.dropout,
            nodal_attention   = args.nodal_attention,
            no_cuda           = args.no_cuda,
            spk2idx           = spk2idx
        )
        model.init_pretrained_embeddings(glv_matrix)
        print('DialogueGCN sur MORI avec', args.base_model)
    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
        else:
            raise NotImplementedError("base model must be GRU, LSTM or None")
        print(f'Basic {args.base_model} Model sur MORI')

    model.to(device)
    loss_function = nn.NLLLoss() if args.graph_model else MaskedNLLLoss()
    optimizer     = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # Training loop
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        def run_epoch(loader, train=False):
            model.train() if train else model.eval()
            total_loss, total_preds, total_labels, total_cids = [], [], [], []

            for batch in loader:
                textf, qmask, umask, emo_labels, maxlens, cids = batch
                lengths = [int(umask[b].sum().item()) for b in range(len(umask))]

                speakers_batch = [data['speakers'][cid] for cid in cids]

                textf = textf.to(device)
                qmask = qmask.to(device)
                umask = umask.to(device)
                emo_labels = emo_labels.to(device)

                if train:
                    optimizer.zero_grad()

                log_prob, *_ = model(textf, qmask, umask, lengths, speakers_batch)

                flat_labels = emo_labels.view(-1)
                flat_umask = umask.view(-1)
                label = flat_labels[flat_umask == 1]

                loss = loss_function(log_prob, label)
                total_loss.append(loss.item())

                if train:
                    loss.backward()
                    optimizer.step()

                preds = log_prob.argmax(dim=1).cpu().numpy()
                labels = label.cpu().numpy()
                total_preds.extend(preds)
                total_labels.extend(labels)
                total_cids.extend(cids)

            acc = accuracy_score(total_labels, total_preds) * 100
            f1 = f1_score(total_labels, total_preds, average='micro') * 100
            return np.mean(total_loss), acc, f1

        tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, train=True)
        va_loss, va_acc, va_f1 = run_epoch(valid_loader, train=False)
        te_loss, te_acc, te_f1 = run_epoch(test_loader, train=False)

        print(f"[Epoch {epoch}/{args.epochs}] "
            f"tr_loss={tr_loss:.4f}, tr_acc={tr_acc:.2f}%, tr_f1={tr_f1:.2f}% | "
            f"va_loss={va_loss:.4f}, va_acc={va_acc:.2f}%, va_f1={va_f1:.2f}% | "
            f"te_loss={te_loss:.4f}, te_acc={te_acc:.2f}%, te_f1={te_f1:.2f}% | "
            f"time={(time.time() - start):.1f}s")

        if te_f1 > best_f1:
            best_f1 = te_f1
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))
            print(f"→ Nouveau meilleur F1: {best_f1:.2f}%, modèle sauvegardé.")

    print("Entraînement terminé.")
    torch.save(model.state_dict(), os.path.join(args.save_path, 'final_model.pt'))
    print("Modèle final enregistré.")



