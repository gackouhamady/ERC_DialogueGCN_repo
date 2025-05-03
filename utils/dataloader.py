import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from torch.nn.utils.rnn import pad_sequence # type: ignore
import pickle, pandas as pd # type: ignore
import os


class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('../data/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]

# utils/dataloader.py

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class MELDDataset(Dataset):
    """
    Dataset PyTorch pour MELD + utilitaires pour baseline_meld.py (Keras).
    
    Usage PyTorch:
        ds = MELDDataset(mode='emotion', train=True)
        loader = DataLoader(ds, batch_size=..., collate_fn=ds.collate_fn, ...)
    
    Usage baseline_meld.py:
        ds = MELDDataset(mode='emotion')
        ds.load_text_data()        # ou load_audio_data(), load_bimodal_data()
        # puis ds.train_dialogue_features, ds.train_dialogue_label, ds.train_mask, ds.train_dialogue_ids, etc.
    """
    def __init__(self,
                 path: str = None,
                 mode: str = None,
                 classify: str = None,
                 train: bool = True):
        # 1) détermination du mode ('emotion' ou 'sentiment')
        if mode is not None:
            classify = mode.lower()
        if classify is None:
            raise ValueError("Vous devez passer `mode=` ou `classify=` ('emotion' ou 'sentiment').")
        classify = classify.lower()
        if classify not in ('emotion', 'sentiment'):
            raise ValueError("`mode` ou `classify` doit valoir 'emotion' ou 'sentiment'.")
        self.classify = classify

        # 2) construction du chemin par défaut
        if path is None:
            base = os.path.dirname(__file__)
            pickles_dir = os.path.abspath(os.path.join(base, '..', 'data', 'MELD_features'))
            filename = f"MELD_{self.classify}.pkl"
            path = os.path.join(pickles_dir, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Impossible de trouver le fichier pickle MELD: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Pas la permission de lire le fichier pickle MELD: {path}")

        # 3) chargement du pickle, support anciens/nouveaux formats
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # ancien format à 9 éléments
        if isinstance(data, (list, tuple)) and len(data) == 9:
            (self.videoIDs,
             self.videoSpeakers,
             self.videoLabelsEmotion,
             self.videoText,
             self.videoAudio,
             self.videoSentence,
             self.trainVid,
             self.testVid,
             self.videoLabelsSentiment) = data
        # nouveau format à 4 éléments
        elif isinstance(data, (list, tuple)) and len(data) == 4:
            (self.videoIDs,
             self.videoText,
             self.videoAudio,
             self.videoSpeakers) = data
            # placeholders pour compatibilité
            self.videoLabelsEmotion = {}
            self.videoSentence = {}
            self.trainVid = []
            self.testVid = []
            self.videoLabelsSentiment = {}
        else:
            raise ValueError(f"Pickle MELD inattendu : {len(data)} éléments (attendu 9 ou 4).")

        # 4) sélection des étiquettes pour le mode choisi
        self.videoLabels = (self.videoLabelsEmotion
                            if self.classify == 'emotion'
                            else self.videoLabelsSentiment)

        # Pour PyTorch DataLoader
        self.keys = list(self.trainVid if train else self.testVid)
        self.len  = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.videoText[vid]),         # [seq_len, text_feat]
            torch.FloatTensor(self.videoAudio[vid]),        # [seq_len, audio_feat]
            torch.FloatTensor(self.videoSpeakers[vid]),     # [seq_len, speaker_feat]
            torch.FloatTensor([1] * len(self.videoLabels[vid])),  # mask
            torch.LongTensor(self.videoLabels[vid]),        # [seq_len]
            vid
        )

    def collate_fn(self, batch):
        # batch : liste de tuples __getitem__
        df = pd.DataFrame(batch)
        out = []
        for i in range(len(df)):
            if i < 3:
                # text, audio, speakers → pad_sequence(time-major)
                out.append(pad_sequence(df[i]))
            elif i < 5:
                # mask, labels → pad_sequence(batch first)
                out.append(pad_sequence(df[i], batch_first=True))
            else:
                # identifiants
                out.append(df[i].tolist())
        return out

    # —————— Méthodes pour baseline_meld.py ——————

    def load_text_data(self,
                       test_size: float = 0.3,
                       val_size: float = 0.5,
                       random_state: int = 42):
        """ Prépare train/val/test sur features textuelles. """
        self._prepare_split(self.videoText,
                            self.videoLabels,
                            test_size, val_size, random_state)

    def load_audio_data(self,
                        test_size: float = 0.3,
                        val_size: float = 0.5,
                        random_state: int = 42):
        """ Prépare train/val/test sur features audio. """
        self._prepare_split(self.videoAudio,
                            self.videoLabels,
                            test_size, val_size, random_state)

    def load_bimodal_data(self,
                          test_size: float = 0.3,
                          val_size: float = 0.5,
                          random_state: int = 42):
        """ Concatène audio+text, puis split. """
        bimodal = {
            vid: np.concatenate([self.videoText[vid],
                                 self.videoAudio[vid]], axis=-1)
            for vid in self.videoIDs
        }
        self._prepare_split(bimodal,
                            self.videoLabels,
                            test_size, val_size, random_state)

    def _prepare_split(self,
                       feat_dict: dict,
                       label_dict: dict,
                       test_size: float,
                       val_size: float,
                       random_state: int):
        # 1) collecte séquences, étiquettes, masques, IDs
        feats, labs, masks, ids = [], [], [], []
        all_labels = []
        for vid in feat_dict:
            x = np.array(feat_dict[vid])    # [seq_len, feat_dim]
            y = np.array(label_dict[vid])   # [seq_len]
            all_labels.extend(y.tolist())
            feats.append(x)
            # one-hot
            num_classes = np.max(all_labels) + 1
            labs.append(np.eye(num_classes)[y])
            masks.append(np.ones(len(y), dtype=int))
            ids.append(vid)
        # 2) padding
        maxlen = max(x.shape[0] for x in feats)
        X = np.stack([np.pad(x, ((0, maxlen-x.shape[0]), (0,0)), 'constant') for x in feats])
        Y = np.stack([np.pad(l, ((0, maxlen-l.shape[0]), (0,0)), 'constant') for l in labs])
        M = np.stack([np.pad(m, (0, maxlen-m.shape[0]), 'constant') for m in masks])
        # 3) split train vs temp
        X_tr, X_temp, Y_tr, Y_temp, M_tr, M_temp, id_tr, id_temp = \
            train_test_split(X, Y, M, ids,
                             test_size=test_size,
                             random_state=random_state)
        # 4) split val vs test
        X_val, X_te, Y_val, Y_te, M_val, M_te, id_val, id_te = \
            train_test_split(X_temp, Y_temp, M_temp, id_temp,
                             test_size=val_size,
                             random_state=random_state)
        # 5) assignation aux attributs
        self.train_dialogue_features = X_tr
        self.val_dialogue_features   = X_val
        self.test_dialogue_features  = X_te

        self.train_dialogue_label = Y_tr
        self.val_dialogue_label   = Y_val
        self.test_dialogue_label  = Y_te

        self.train_mask = M_tr
        self.val_mask   = M_val
        self.test_mask  = M_te

        # dictionnaires ID→index (préserve l’ordre)
        self.train_dialogue_ids = {vid: idx for idx, vid in enumerate(id_tr)}
        self.val_dialogue_ids   = {vid: idx for idx, vid in enumerate(id_val)}
        self.test_dialogue_ids  = {vid: idx for idx, vid in enumerate(id_te)}


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class DailyDialogueDataset2(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.FloatTensor(list(self.Features[conv])), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]

class DailyDialogueDataset3(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.Features, _, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.FloatTensor(self.Features[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]

# no use just copy from DialogueRNN
class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i==0 else \
                pad_sequence(dat[i]) if i == 1 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]