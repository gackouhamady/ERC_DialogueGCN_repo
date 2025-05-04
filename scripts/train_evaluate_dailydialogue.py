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

        self.Speakers, self.Features, self.InputMaxSequenceLength, \
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

        return torch.LongTensor(self.Features[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               self.InputMaxSequenceLength[conv], \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]






def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_DailyDialogue_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    trainset = DailyDialogueDataset2('train', path)
    testset = DailyDialogueDataset2('test', path)
    validset = DailyDialogueDataset2('valid', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

# --- en tout début (après parser.parse_args()) ---
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f"Running on device: {device}")

def process_data_loader(data):
    textf, qmask, umask, label, max_sequence_lengths, _ = data
    # tronquer au max length du batch
    input_sequence = textf[:, :, : max(max_sequence_lengths)]

    # Envoi sur le bon device
    input_sequence     = input_sequence.to(device)
    qmask              = qmask.to(device)
    umask              = umask.to(device)
    emotion_labels     = label.to(device)

    return [input_sequence, qmask, umask, emotion_labels]



import time
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)

def train_or_eval_graph_model(
    model,
    loss_function,
    dataloader,
    epoch,
    cuda,
    optimizer=None,
    train=False
):
    device = 'cuda' if cuda else 'cpu'
    model.train() if train else model.eval()

    all_losses, all_preds, all_labels, all_vids = [], [], [], []
    ei = torch.empty(0, device=device)
    et = torch.empty(0, device=device)
    en = torch.empty(0, device=device)
    el = []

    for step, data in enumerate(dataloader):
        # … (1) Unpacking et (2)-(6) identiques à votre version …
        textf_raw, qmask_raw, umask_raw, label_raw, lengths_raw, vid_raw = data
        if train: optimizer.zero_grad()
        textf, qmask, umask, label_list, *rest = process_data_loader(data)
        textf, qmask, umask = textf.to(device), qmask.to(device), umask.to(device)
        B, T = umask.size()
        lengths = [int(umask[b].sum().item()) for b in range(B)]
        log_prob, e_i_b, e_n_b, e_t_b, e_l_b = model(textf, qmask, umask, lengths)
        N_pred, C = log_prob.shape

        # … (7) Flatten labels & mask, même ordre que le modèle …
        if isinstance(label_list, torch.Tensor):
            flat_labels = label_list.contiguous().view(-1)
        else:
            lab = torch.stack(label_list, dim=0).permute(1,0).contiguous()  # [B,T]
            flat_labels = lab.view(-1)
        flat_umask = umask.contiguous().view(-1)  # [B*T]

        total_pos = flat_umask.numel()
        kept_pos  = int(flat_umask.sum().item())

        print(f"[Batch {step}] pre-align: preds={N_pred}, kept={kept_pos}")

        # === (8) NOUVELLE BOUCLE DE REPAD/TRIM ===
        # On réitère jusqu’à ce que preds == labels
        while N_pred != kept_pos:
            diff = kept_pos - N_pred
            if diff > 0:
                # il manque des prédictions → on répète la dernière ligne
                print(f"  [PAD LOG] manque {diff}, on répète la dernière sortie du modèle")
                last = log_prob[-1:].detach()             # (1, C)
                pad  = last.expand(diff, C)               # (diff, C)
                log_prob = torch.cat([log_prob, pad], 0)
            else:
                # trop de prédictions → on enlève des labels excédentaires
                cut = -diff
                print(f"  [TRIM LAB] {cut} labels excédentaires, on les coupe")
                # on trouve les indices des positions valides
                idxs = (flat_umask == 1).nonzero(as_tuple=False).squeeze(1)
                # on retire les 'cut' derniers de ces idxs
                drop = idxs[-cut:]
                mask_keep = torch.ones_like(flat_umask, dtype=torch.bool)
                mask_keep[drop] = False
                flat_labels = flat_labels[mask_keep]
                flat_umask  = flat_umask[mask_keep]

            # recalc
            N_pred  = log_prob.size(0)
            kept_pos = int(flat_umask.sum().item())
            print(f"    → now preds={N_pred}, kept={kept_pos}")

        # (9) Slice final
        label = flat_labels[flat_umask == 1]
        assert label.numel() == N_pred, "Alignement final échoué"

        # (10)-(13) Loss, backward, collect, logs …
        loss = loss_function(log_prob, label.to(device))
        all_losses.append(loss.item())
        ei = torch.cat([ei, e_i_b], dim=1)
        et = torch.cat([et, e_t_b], dim=0)
        en = torch.cat([en, e_n_b], dim=0)
        el += e_l_b
        all_preds.append(log_prob.argmax(dim=1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_vids.extend(vid_raw)

        if train:
            loss.backward()
            optimizer.step()
        if train and step % 10 == 0:
            ps = np.concatenate(all_preds)
            ls = np.concatenate(all_labels)
            print(f" Step {step} | loss={np.mean(all_losses):.4f} | "
                  f"acc={accuracy_score(ls,ps)*100:.2f}% | "
                  f"f1={f1_score(ls,ps,average='micro')*100:.2f}%")

    # (14) Agrégation finale identique …
    if not all_preds:
        return [float('nan')] * 12
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    avg_loss = float(np.mean(all_losses))
    avg_acc  = accuracy_score(labels, preds) * 100
    avg_f1   = f1_score(labels, preds, average='micro') * 100
    avg_prec = precision_score(labels, preds, average='micro') * 100
    avg_rec  = recall_score(labels, preds, average='micro') * 100

    return (
        avg_loss, avg_acc,
        labels, preds, avg_f1,
        np.array(all_vids),
        ei.cpu().numpy(), et.cpu().numpy(),
        en.detach().cpu().numpy(), np.array(el),

        avg_prec, avg_rec
    )






if __name__ == '__main__':

    path = './saved/DailyDialog/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--save_path', type=str, default='./saved/dialogGCN_{}'.format(
        dt.datetime.now().strftime("%m_%d_%H_%M")),
                        metavar='save_path', help='model save path')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_classes = 7
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    # change D_m into
    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100
    kernel_sizes = [3, 4, 5]
    glv_pretrained = np.load(open('../data/dailydialog/glv_embedding_matrix', 'rb'), allow_pickle=True)
    vocab_size, embedding_dim = glv_pretrained.shape
    if args.graph_model:
        seed_everything()

        model = DialogueGCN_DailyModel(args.base_model,
                                       D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                       n_speakers=2,
                                       max_seq_len=110,
                                       window_past=args.windowp,
                                       window_future=args.windowf,
                                       vocab_size=vocab_size,
                                       n_classes=n_classes,
                                       listener_state=args.active_listener,
                                       context_attention=args.attention,
                                       dropout=args.dropout,
                                       nodal_attention=args.nodal_attention,
                                       no_cuda=args.no_cuda
                                       )
        model.init_pretrained_embeddings(glv_pretrained)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()


    if args.graph_model:
        loss_function = nn.NLLLoss()
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_DailyDialogue_loaders('../data/dailydialog/daily_dialogue2.pkl',
                                                                        batch_size=batch_size, num_workers=0)
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    all_precision, all_recall = [], []

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, train_precision, train_recall = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _, valid_precision, valid_recall = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_precision, test_recall = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda)
            all_fscore.append(test_fscore)
            all_precision.append(test_precision)
            all_recall.append(test_recall)
            # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')


        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, # type: ignore
                                                                                  optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e) # type: ignore
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, # type: ignore
                                                                                                                 loss_function,
                                                                                                                 test_loader,
                                                                                                                 e)
            all_fscore.append(test_fscore)
            # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, train_precision: {}, train_recall: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, valid_precision: {}, valid_recall: {}, test_loss: {}, test_acc: {}, test_fscore: {}, test_precision: {}, test_recall: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, train_precision, train_recall, valid_loss, valid_acc,
                   valid_fscore, valid_precision, valid_recall, test_loss, test_acc, test_fscore, test_precision,
                   test_recall, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    # save the model for latter
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pkl'))
    print('Saved model for latter utilization.')
