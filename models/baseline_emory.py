import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math

# For methods and models related to DialogueGCN jump to line 516





class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        #score = F.softmax(score, dim=-1)
        score = F.softmax(score, dim=0)
        # print (score)
        # print (sum(score))
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedEdgeAttention(nn.Module):
    """
    Compute edge weights as in Equation (1) of DialogueGCN.
    Supports three variants: 'attn1' (default), 'attn2', 'attn3'.
    """
    def __init__(self, input_dim, max_seq_len):
        super(MaskedEdgeAttention, self).__init__()
        self.input_dim   = input_dim
        self.max_seq_len = max_seq_len
        # scalar: projects each vector to a score over sequence positions
        self.scalar     = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.simpleatt  = SimpleAttention(self.input_dim)
        self.att        = Attention(self.input_dim, score_function='mlp')
        # MatchingAttention is only used in variants you may implement
        self.matchatt   = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')

    def forward(self, M, lengths, edge_ind):
        """
        Args:
            M         (Tensor): shape (seq_len, batch, input_dim)
            lengths   (List[int]): actual lengths per dialogue in the batch
            edge_ind  (List[List[Tuple[int,int]]]): for each dialogue j, list of (u,v) edges
        Returns:
            scores   (Tensor): shape (batch, max_seq_len, max_seq_len)
        """
        device     = M.device
        batch_size = M.size(1)
        attn_type  = 'attn1'

        if attn_type == 'attn1':
            # ---- Compute full attention matrix ----
            # scale: (seq_len, batch, max_seq_len)
            scale = self.scalar(M)
            # alpha: (batch, max_seq_len, seq_len)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)

            # create masks on the correct device
            mask      = torch.ones_like(alpha, device=device) * 1e-10
            mask_copy = torch.zeros_like(alpha, device=device)

            # build index list, filter out-of-bounds
            idx_list = []
            for j, perms in enumerate(edge_ind):
                for u, v in perms:
                    if u < self.max_seq_len and v < self.max_seq_len:
                        idx_list.append((j, u, v))
            if idx_list:
                idx = torch.tensor(idx_list, dtype=torch.long, device=device).t()
                # allow only edges in edge_ind
                mask[idx[0], idx[1], idx[2]]      = 1.0
                mask_copy[idx[0], idx[1], idx[2]] = 1.0

            masked_alpha = alpha * mask
            sums         = masked_alpha.sum(-1, keepdim=True)
            scores       = masked_alpha.div(sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            # initialize scores on correct device
            scores = torch.zeros(batch_size, self.max_seq_len, self.max_seq_len, device=device)
            for j, perms in enumerate(edge_ind):
                for u, v in perms:
                    if u < self.max_seq_len and v < self.max_seq_len:
                        M_nei    = M[v, j, :].unsqueeze(0).unsqueeze(1)
                        M_u      = M[u, j, :].unsqueeze(0).unsqueeze(0)
                        _, alpha_uv = self.simpleatt(M_nei, M_u)
                        scores[j, u, v] = alpha_uv.view(-1)
            return scores

        elif attn_type == 'attn3':
            scores = torch.zeros(batch_size, self.max_seq_len, self.max_seq_len, device=device)
            for j, perms in enumerate(edge_ind):
                for u, v in perms:
                    if u < self.max_seq_len and v < self.max_seq_len:
                        M_nei      = M[v, j, :].unsqueeze(0).unsqueeze(1)
                        M_u        = M[u, j, :].unsqueeze(0).unsqueeze(0)
                        _, alpha_uv = self.att(M_nei, M_u)
                        scores[j, u, v] = alpha_uv.view(-1)
            return scores

        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")



def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    """

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        # 取窗口内的句子
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        # 构造句子和句子的关系 perms <class 'set'>: {(1, 2), (1, 3), (1, 4), (1, 0), (1, 1)}
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)      # 只保留唯一的值
    return list(all_perms)
    
# --- en tout début (après parser.parse_args()) ---
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f"Running on device: {device}")
import torch # type: ignore


def batch_graphify(
    features,            # Tensor [T,B,D] ou [B,T,D]
    qmask,               # Tensor [T,B,S] ou [B,T,S]
    lengths,             # List[int], len == batch_size
    window_past, 
    window_future,
    edge_type_mapping,   # dict[str,int], clés comme "i k rel" ex "010"
    spk2idx,             # dict[str,int], mapping nom_de_speaker → petit entier
    speakers,            # List[List[str]], shape (B, ≥T) liste des noms par dialogue
    att_model,           # module renvoyant scores [B,T,T]
    device,
    max_edge_speakers=2  # pour limiter la combinatoire des types de relation
):
    B = len(lengths)
    if features.size(0) == B:
        features = features.permute(1, 0, 2).contiguous()
        print("[bg] permuted features BTD→TBD")
    T, Bf, D = features.size()
    assert Bf == B, f"[bg] Bf != B ({Bf} != {B})"

    if qmask.size(0) == B:
        qmask = qmask.permute(1, 0, 2).contiguous()
        print("[bg] permuted qmask BTS→TBS")
    _, Bm, S = qmask.size()
    assert Bm == B, f"[bg] Bm != B ({Bm} != {B})"

    # Clamp lengths
    lengths = [max(0, min(l, T)) for l in lengths]

    edge_perms_list = [edge_perms(lengths[j], window_past, window_future) for j in range(B)]
    scores = att_model(features, lengths, edge_perms_list)
    print(f"[bg] scores.shape = {scores.shape}")

    node_feats, idx_list, norm_list, type_list = [], [], [], []
    edge_index_lengths = []
    offset = 0

    for j, perms in enumerate(edge_perms_list):
        L = lengths[j]
        if L > 0:
            node_feats.append(features[:L, j, :].to(device))
        edge_index_lengths.append(len(perms))

        spk_names = speakers[j]
        if len(spk_names) < L:
            raise ValueError(f"[bg] speakers[{j}] length {len(spk_names)} < {L}")

        for (u, v) in perms:
            if u >= L or v >= L: continue

            su, sv = u + offset, v + offset
            idx_list.append([su, sv])
            norm_list.append(scores[j, u, v].to(device))

            i_raw = spk2idx.get(spk_names[u], 0)
            k_raw = spk2idx.get(spk_names[v], 0)
            i = i_raw % max_edge_speakers
            k = k_raw % max_edge_speakers
            rel = '0' if u < v else '1'
            et_key = f"{i}{k}{rel}"

            if et_key not in edge_type_mapping:
                print(f"[bg][WARN] missing edge type: {et_key}, using fallback 0")
                type_list.append(0)
            else:
                type_list.append(edge_type_mapping[et_key])
        offset += L

    node_features = torch.cat(node_feats, dim=0) if node_feats else torch.empty((0, D), device=device)
    edge_index = torch.tensor(idx_list, dtype=torch.long, device=device).t().contiguous() if idx_list else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_norm = torch.stack(norm_list).to(device) if norm_list else torch.empty((0,), dtype=features.dtype, device=device)
    edge_type = torch.tensor(type_list, dtype=torch.long, device=device) if type_list else torch.empty((0,), dtype=torch.long, device=device)

    print(f"[bg] out: nodes={node_features.size(0)}, edges={edge_index.size(1)}")
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths



def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """
    
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    
    #if torch.cuda.is_available():
    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda) 
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)


    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:,0,:])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, linear_layer, linear_beta, dropout_layer, smax_fc_layer, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    """
    before = linear_beta(emotions)
    late = torch.mm(before, emotions.T)
    beta = F.softmax(late)
    emotions = torch.mm(beta, emotions)
    hidden = F.relu(linear_layer(emotions))
    hidden = dropout_layer(hidden)
    hidden = smax_fc_layer(hidden)

    log_prob = F.log_softmax(hidden, 1)
    return log_prob




import torch.nn as nn # type: ignore
from torch_geometric.nn import RGCNConv, GraphConv # type: ignore

class GraphNetwork_MORI(nn.Module):
    """
    GCN reasoning module adapté pour MORI NLP (DialogueGCN variant).
    2 couches RGCN + GraphConv, suivi de matching attention et classification node-wise.
    """
    def __init__(
        self,
        in_feats: int,
        n_classes: int,
        n_relations: int,
        max_seq_len: int,
        graph_hid_dim: int = 64,
        dropout: float = 0.5,
        no_cuda: bool = False
    ):
        super().__init__()
        self.in_feats      = in_feats
        self.n_classes     = n_classes
        self.n_relations   = n_relations
        self.max_seq_len   = max_seq_len
        self.graph_hid_dim = graph_hid_dim
        self.no_cuda       = no_cuda

        # 1) Couche RGCN
        self.conv1 = RGCNConv(
            in_channels   = in_feats,
            out_channels  = graph_hid_dim,
            num_relations = n_relations,
            num_bases     = min(n_relations, 30)
        )

        # 2) Couche GraphConv
        self.conv2 = GraphConv(
            in_channels  = graph_hid_dim,
            out_channels = graph_hid_dim
        )

        # 3) Matching Attention
        att_dim = in_feats + graph_hid_dim
        self.matchatt = MatchingAttention(
            mem_dim  = att_dim,
            cand_dim = att_dim,
            alpha_dim=None,
            att_type ='general2'
        )

        # 4) Layers de classification
        self.linear      = nn.Linear(in_feats + graph_hid_dim, graph_hid_dim)
        self.linear_beta = nn.Linear(in_feats + graph_hid_dim, in_feats + graph_hid_dim)
        self.dropout     = nn.Dropout(dropout)
        self.smax_fc     = nn.Linear(graph_hid_dim, n_classes)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_norm: torch.Tensor,
                edge_type: torch.LongTensor,
                seq_lengths: list[int],
                umask: torch.Tensor
    ) -> torch.Tensor:
        # 1) RGCN
        h1 = self.conv1(x, edge_index, edge_type)
        # 2) GraphConv
        h2 = self.conv2(h1, edge_index)
        # 3) Fusion features
        h = torch.cat([x, h2], dim=-1)
        # 4) Attention + classification
        log_prob = classify_node_features(
            h,
            self.linear,
            self.linear_beta,
            self.dropout,
            self.smax_fc,
            self.no_cuda
        )
        return log_prob



class DialogueGCNModel(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, no_cuda=False):
        
        super(DialogueGCNModel, self).__init__()

        self.base_model = base_model
        self.no_cuda = no_cuda

        # The base model is the sequential context encoder.
        if self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 

        n_relations = n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)

        self.graph_net = GraphNetwork_MORI(2*D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping


    def forward(self, U, qmask, umask, seq_lengths, speakers_list):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        if self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

            # Après, en passant aussi spk2idx et la liste des speakers pour le batch, et le device :
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            emotions,                          # [T, B, D]
            qmask,                             # [T, B, S]
            seq_lengths,                       # List[int]
            self.window_past,                  # int
            self.window_future,                # int
            self.edge_type_mapping,            # dict[str,int]
            self.spk2idx,                      # dict[speaker_name -> idx]
            speakers_list,                     # List[List[str]] pour le batch (passed as argument)
            self.att_model,                    # module d’attention
            self.device                        # torch.device
        )

        log_prob = self.graph_net(
            features,
            edge_index,
            edge_norm,
            edge_type,
            seq_lengths,
            umask.to(self.device)
        )

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths


# class CNNFeatureExtractor(nn.Module):
#     """
#     Module from DialogueRNN
#     """
#     def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
#         super(CNNFeatureExtractor, self).__init__()
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.convs = nn.ModuleList(
#             [nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
#         self.feature_dim = output_size
#
#     def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
#         self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
#         # if is_static:
#         self.embedding.weight.requires_grad = False
#
#     def forward(self, x, umask):
#         num_utt, batch, num_words = x.size()
#
#         x = x.type(LongTensor)  # (num_utt, batch, num_words)
#         x = x.view(-1, num_words)  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
#         emb = self.embedding(x)  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
#         emb = emb.transpose(-2,
#                             -1).contiguous()  # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)
#
#         convoluted = [F.relu(conv(emb)) for conv in self.convs]
#         pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
#         concated = torch.cat(pooled, 1)
#         features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
#         features = features.view(num_utt, batch, -1)  # (num_utt * batch, 100) -> (num_utt, batch, 100)
#         mask = umask.unsqueeze(-1).type(FloatTensor)  # (batch, num_utt) -> (batch, num_utt, 1)
#         mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
#         mask = mask.repeat(1, 1, self.feature_dim)  # (num_utt, batch, 1) -> (num_utt, batch, 100)
#         features = (features * mask)  # (num_utt, batch, 100) -> (num_utt, batch, 100)
#
#         return features

import torch
import torch.nn as nn
import numpy as np



class DialogueGCN_MORIModel(nn.Module):
    """
    DialogueGCN adapté pour MORI NLP :
      - encodage phrase par phrase (D_m)
      - projection “party” (D_p)
      - séquence d’énoncés (D_e)
      - attention nodale (D_a)
      - raisonnement GCN (graph_hidden_size)
      - classification node-wise en n_classes
    """
    def __init__(
        self,
        base_model: str,
        D_m: int,
        D_p: int,
        D_e: int,
        D_h: int,
        D_a: int,
        graph_hidden_size: int,
        n_classes: int,
        n_speakers: int,
        max_seq_len: int,
        window_past: int,
        window_future: int,
        vocab_size: int,
        embedding_dim: int = 300,
        dropout: float = 0.5,
        nodal_attention: bool = True,
        no_cuda: bool = False,
        spk2idx: dict = None,               # ← mapping interne nom→idx
    ):
        super().__init__()
        # 0) Device
        self.device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

        # 1) Stockage hyperparamètres
        self.D_m = D_m
        self.D_p = D_p   # “party” projection (DB)
        self.D_e = D_e
        self.D_a = D_a
        self.window_past   = window_past
        self.window_future = window_future
        self.spk2idx       = spk2idx or {}
        self.nodal_attention = nodal_attention

        # 2) Embedding + LSTM sentence
        self.embedding_dim = embedding_dim
        self.embedding     = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.lstm_sen      = nn.LSTM(
            input_size  = embedding_dim,
            hidden_size = D_m,
            num_layers  = 2,
            bidirectional=False,
            dropout      = dropout
        ).to(self.device)

        # 3) Séquence d’énoncés
        self.base_model = base_model
        if base_model == 'LSTM':
            self.lstm = nn.LSTM(
                input_size  = D_m,
                hidden_size = D_e,
                num_layers  = 2,
                bidirectional=True,
                dropout      = dropout
            ).to(self.device)
        elif base_model == 'GRU':
            self.gru = nn.GRU(
                input_size  = D_m,
                hidden_size = D_e,
                num_layers  = 2,
                bidirectional=True,
                dropout      = dropout
            ).to(self.device)
        elif base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e).to(self.device)
        else:
            raise NotImplementedError(f"Unknown base_model: {base_model}")

        # 4) Attention nodale

        n_relations      = n_speakers**2
        self.att_model   = MaskedEdgeAttention(2 * D_e, max_seq_len).to(self.device)

        # 5) Raisonnement GCN + classification
      
        self.graph_net = GraphNetwork_MORI(
            in_feats      = 2 * D_e,
            n_classes     = n_classes,
            n_relations   = n_relations,
            max_seq_len   = max_seq_len,
            graph_hid_dim = graph_hidden_size,
            dropout       = dropout,
            no_cuda       = no_cuda
        ).to(self.device)

        # 6) Mapping types d’arête
       
        
        self.max_edge_speakers = 2  # Fixé pour limiter la complexité à 2 speakers (→ 2*2*2 = 8 relations max)
        edge_type_mapping = {}
        for i in range(self.max_edge_speakers):
            for j in range(self.max_edge_speakers):
                edge_type_mapping[f"{i}{j}0"] = len(edge_type_mapping)  # forward
                edge_type_mapping[f"{i}{j}1"] = len(edge_type_mapping)  # backward
        self.edge_type_mapping = edge_type_mapping

    def init_pretrained_embeddings(self, pretrained_vectors):
        """Charge et fige GloVe."""
        if isinstance(pretrained_vectors, np.ndarray):
            pretrained_vectors = torch.from_numpy(pretrained_vectors)
        pretrained_vectors = pretrained_vectors.to(self.device)
        with torch.no_grad():
            self.embedding.weight.data.copy_(pretrained_vectors)
            self.embedding.weight.requires_grad = False

    def forward(self, input_seq, qmask, umask, seq_lengths, batch_speakers):
        """
        Args:
          input_seq      : LongTensor [B, U, L]
          qmask          : FloatTensor [B, U, S] (locuteur→relation)
          umask          : FloatTensor [B, U]   (masque énoncés)
          seq_lengths    : List[int]
          batch_speakers : FloatTensor [B, U, S] one-hot locuteurs
        Returns:
          log_prob : [N_nodes, n_classes]
          edge_index, edge_norm, edge_type, ei_len
        """
        B, U, L = input_seq.size()
        # — Phrase encoding
        x = self.embedding(input_seq.to(self.device))  # [B,U,L,emb]
        x = x.permute(2,0,1,3).reshape(L, B*U, self.embedding_dim)
        Ut, _ = self.lstm_sen(x)
        U_last = Ut[-1].view(B, U, self.D_m)

        # — Sequence encoding
        U_in = U_last.permute(1,0,2)  # [U,B,D_m]
        if self.base_model == 'LSTM':
            emotions, _ = self.lstm(U_in)
        elif self.base_model == 'GRU':
            emotions, _ = self.gru(U_in)
        else:
            emotions    = self.base_linear(U_in)

        # — Build big graph
    
        # qmask doit être transposé en [T,B,S]
        qm = qmask.permute(1,0,2).contiguous()
            # 3) Construction du graphe (avec spk2idx + liste des speakers)
        features, edge_index, edge_norm, edge_type, ei_len = batch_graphify(
            emotions,                     # Tensor [T, B, 2*D_e]
            qmask.to(self.device),        # Tensor [T, B, n_speakers]
            seq_lengths,                  # List[int] longueurs réelles par dialogue
            self.window_past,             # int
            self.window_future,           # int
            self.edge_type_mapping,       # dict[str,int]
            self.spk2idx,                 # dict[name→idx]
            batch_speakers,               # List[List[str]] des noms de speakers, len=B
            self.att_model,               # module d’attention renvoyant [B,T,T]
            self.device,
            max_edge_speakers=self.max_edge_speakers  # ← ajoute ici                   # torch.device
        )

        # 4) GCN + classification
        log_prob = self.graph_net(
            features, 
            edge_index, 
            edge_norm, 
            edge_type, 
            seq_lengths, 
            umask.to(self.device)         # Tensor [B, U]
        )
        return log_prob, edge_index, edge_norm, edge_type, ei_len

  
