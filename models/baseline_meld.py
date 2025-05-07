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
        self.scalar   = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att       = Attention(self.input_dim, score_function='mlp')
        # MatchingAttention is only used in variants you may implement
        self.matchatt  = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')

    def forward(self, M, lengths, edge_ind):
        """
        Args:
            M         (Tensor): shape (seq_len, batch, input_dim)
            lengths   (List[int]): actual lengths per dialogue in the batch
            edge_ind  (List[List[Tuple[int,int]]]): for each dialogue j, list of (u,v) edges
        Returns:
            scores   (Tensor): shape (batch, max_seq_len, max_seq_len)
        """
        device = M.device
        batch_size = M.size(1)
        attn_type = 'attn1'

        if attn_type == 'attn1':
            # ---- Compute full attention matrix ----
            # scale: (seq_len, batch, max_seq_len)
            scale = self.scalar(M)  
            # alpha: (batch, max_seq_len, seq_len)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)

            # create masks on the correct device
            mask      = torch.ones_like(alpha, device=device) * 1e-10
            mask_copy = torch.zeros_like(alpha, device=device)

            # build index tensor of shape (3, total_edges)
            idx_list = []
            for j, perms in enumerate(edge_ind):
                for u, v in perms:
                    idx_list.append([j, u, v])
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
            scores = torch.zeros(batch_size, self.max_seq_len, self.max_seq_len,
                                 device=device)
            # fill for each dialogue
            for j in range(batch_size):
                perms = edge_ind[j]
                for u, v in perms:
                    # neighbor embeddings
                    M_nei = M[v, j, :].unsqueeze(0).unsqueeze(1)  # (1,1,dim)
                    M_u   = M[u, j, :].unsqueeze(0).unsqueeze(0)  # (1,1,dim)
                    _, alpha_uv = self.simpleatt(M_nei, M_u)      # (1,1)
                    scores[j, u, v] = alpha_uv.view(-1)
            return scores

        elif attn_type == 'attn3':
            scores = torch.zeros(batch_size, self.max_seq_len, self.max_seq_len,
                                 device=device)
            for j in range(batch_size):
                perms = edge_ind[j]
                for u, v in perms:
                    M_nei = M[v, j, :].unsqueeze(0).unsqueeze(1)   # (1,1,dim)
                    M_u   = M[u, j, :].unsqueeze(0).unsqueeze(0)   # (1,1,dim)
                    # Attention returns (context, weights)
                    _, alpha_uv = self.att(M_nei, M_u)            # (1,1,1)
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
import torch


def batch_graphify(
    features,            # Tensor [T,B,D] ou [B,T,D]
    qmask,               # Tensor [T,B,S] ou [B,T,S]
    lengths,             # List[int], attendu len==batch_size
    window_past, 
    window_future,
    edge_type_mapping,   # dict[str,int]
    att_model,           # module renvoyant scores [B,T,T]
    device
):
    """
    Construit le big-graph PyG pour le batch.
    Retourne :
      node_features:      Tensor [N_nodes, D]
      edge_index:         LongTensor [2, N_edges]
      edge_norm:          Tensor [N_edges]
      edge_type:          LongTensor [N_edges]
      edge_index_lengths: List[int]
    """
    f, m = features, qmask
    B = len(lengths)

    # 1) DETECTION / PERMUTATION features -------------
    # si f.shape = (B,T,D) → permute en (T,B,D)
    if f.dim() != 3:
        raise ValueError(f"[bg] features doit être 3D, got {f.dim()}D")
    if f.size(1) == B:
        # already [T,B,D]
        T, B_f, D = f.size()
    elif f.size(0) == B:
        # [B,T,D] → permute
        f = f.permute(1,0,2).contiguous()
        T, B_f, D = f.size()
        print(f"[bg] permuted features BTD→TBD")
    else:
        raise ValueError(f"[bg] features dims incompatible {tuple(f.shape)} for batch_size={B}")

    if B_f != B:
        raise ValueError(f"[bg] batch_size mismatch: features B={B_f} vs lengths={B}")

    # 2) DETECTION / PERMUTATION qmask ---------------
    if m.dim() != 3:
        raise ValueError(f"[bg] qmask doit être 3D, got {m.dim()}D")
    if m.size(1) == B:
        # already [T,B,S]
        _, B_m, S = m.size()
    elif m.size(0) == B:
        # [B,T,S] → permute
        m = m.permute(1,0,2).contiguous()
        _, B_m, S = m.size()
        print(f"[bg] permuted qmask BTS→TBS")
    else:
        raise ValueError(f"[bg] qmask dims incompatible {tuple(m.shape)} for batch_size={B}")

    if B_m != B:
        raise ValueError(f"[bg] batch_size mismatch: qmask B={B_m} vs lengths={B}")

    # 3) ALIGN lengths                              
    if len(lengths) != B:
        raise ValueError(f"[bg] lengths list must have exactly batch_size={B}")
    # clamp lengths to [0,T]
    for i, L in enumerate(lengths):
        if L < 0 or L > T:
            new = max(0, min(L, T))
            print(f"[bg] clamped lengths[{i}] {L}→{new}")
            lengths[i] = new

    # 4) BUILD edge_perms & SCORES                  
    edge_perms_list = [
        edge_perms(lengths[j], window_past, window_future)
        for j in range(B)
    ]
    scores = att_model(f, lengths, edge_perms_list)  # → [B, T, T]
    print(f"[bg] scores.shape = {tuple(scores.shape)}")

    # 5) ASSEMBLE NODES + EDGES                      
    node_feats, idx_list, norm_list, type_list = [], [], [], []
    edge_index_lengths = []
    offset = 0

    seen_edge_type_defaults = set()
    for j, perms in enumerate(edge_perms_list):
        L = lengths[j]
        # collect nodes
        if L > 0:
            node_feats.append(f[:L, j, :].to(device))
        edge_index_lengths.append(len(perms))

        # collect edges for dialogue j
        for (u, v) in perms:
            uj = min(max(u, 0), T-1)
            vj = min(max(v, 0), T-1)
            bj = j
            su, sv = uj + offset, vj + offset
            idx_list.append([su, sv])
            norm_list.append(scores[bj, uj, vj].to(device))

            # speaker lookup (clamp missing→0)
            mu = m[uj, bj]
            nz = mu.nonzero(as_tuple=False)
            spk_u = int(nz[0,0]) if nz.numel() else 0

            mv = m[vj, bj]
            nz = mv.nonzero(as_tuple=False)
            spk_v = int(nz[0,0]) if nz.numel() else 0

            rel = '0' if uj < vj else '1'
            key = f"{spk_u}{spk_v}{rel}"
            et = edge_type_mapping.get(key)
            if et is None:
                et = 0
                if key not in seen_edge_type_defaults:
                    print(f"[bg] missing edge_type '{key}' → default 0")
                    seen_edge_type_defaults.add(key)
                edge_type_mapping[key] = 0
            type_list.append(et)

        offset += L

    # 6) CONCAT & RETURN                            
    node_features = (torch.cat(node_feats, dim=0) if node_feats
                     else torch.empty((0, D), device=device))
    node_features = node_features.to(device)

    if idx_list:
        edge_index = torch.tensor(idx_list, dtype=torch.long).t().contiguous().to(device)
        edge_norm  = torch.stack(norm_list).to(device)
        edge_type  = torch.tensor(type_list, dtype=torch.long).to(device)
    else:
        edge_index = torch.empty((2,0),dtype=torch.long,device=device)
        edge_norm  = torch.empty((0,),dtype=f.dtype,device=device)
        edge_type  = torch.empty((0,),dtype=torch.long,device=device)

    print(f"[bg] out: nodes={node_features.size(0)} edges={edge_index.size(1)}")
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


import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv

class GraphNetwork(nn.Module):
    """
    GCN reasoning module for Emotion Recognition in Conversation (DialogueGCN).
    Implements a 2-layer RGCN with matching attention and node-wise classification.
    """
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        n_relations: int,
        max_seq_len: int,
        graph_hid_dim: int = 64,
        dropout: float = 0.5,
        no_cuda: bool = False
    ):
        super(GraphNetwork, self).__init__()
        # Paramètres internes
        self.in_feats        = in_feats
        self.out_feats       = out_feats
        self.n_relations     = n_relations
        self.max_seq_len     = max_seq_len
        self.graph_hid_dim   = graph_hid_dim
        self.no_cuda         = no_cuda

        # 1) Première couche RGCN
        self.conv1 = RGCNConv(
            in_channels = in_feats,
            out_channels = graph_hid_dim,
            num_relations = n_relations,
            num_bases = min(n_relations, 30)
        )

        # 2) Deuxième couche GraphConv
        self.conv2 = GraphConv(
            in_channels = graph_hid_dim,
            out_channels = graph_hid_dim
        )

        # 3) Matching Attention
        # Signature: MatchingAttention(mem_dim, cand_dim, alpha_dim=None, att_type='general')
        att_dim = in_feats + graph_hid_dim
        self.matchatt = MatchingAttention(
            mem_dim  = att_dim,
            cand_dim = att_dim,
            alpha_dim=None,
            att_type = 'general2'
        )

        # 4) Couches linéaires pour classification
        self.linear      = nn.Linear(in_feats + graph_hid_dim, graph_hid_dim)
        self.linear_beta = nn.Linear(in_feats + graph_hid_dim, in_feats + graph_hid_dim)
        self.dropout     = nn.Dropout(dropout)
        # Couche de classification finale
        self.smax_fc     = nn.Linear(graph_hid_dim, out_feats)

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask):
        """
        x: Tensor [N, in_feats] Node features
        edge_index: LongTensor [2, E] Edge connectivity
        edge_norm: Tensor [E] Edge normalization (unused here)
        edge_type: LongTensor [E] Edge relation types
        seq_lengths: List[int] Nombre d'énoncés par dialogue
        umask: Tensor [N] Masque des nœuds valides

        Returns:
            log_prob: Tensor [N, out_feats] Log-probabilités par nœud
        """
        # Convolution RGCN
        h1 = self.conv1(x, edge_index, edge_type)
        # Convolution GraphConv
        h2 = self.conv2(h1, edge_index)
        # Concaténation features
        h_concat = torch.cat([x, h2], dim=-1)
        # Classification node-wise
        log_prob = classify_node_features(
            h_concat,
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

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)

        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping


    def forward(self, U, qmask, umask, seq_lengths):
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

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.att_model, self.no_cuda)
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask)

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




class DialogueGCN_MELDModel(nn.Module):
    """
    DialogueGCN adapté pour le dataset MELD:
      - encodage phrase par phrase (LSTM)
      - modèle séquentiel de type LSTM/GRU/linéaire sur les énoncés
      - raisonnement de graphe via GCN
    Charger ensuite ce modèle dans votre pipeline MELD.
    """
    def __init__(
        self,
        base_model: str,
        D_m: int,
        D_g: int,
        D_p: int,
        D_e: int,
        D_h: int,
        D_a: int,
        graph_hidden_size: int,
        n_speakers: int,
        max_seq_len: int,
        window_past: int,
        window_future: int,
        vocab_size: int,
        embedding_dim: int = 300,
        dropout: float = 0.5,
        nodal_attention: bool = True,
        no_cuda: bool = False
    ):
        super().__init__()

        # Stockage des fenêtres pour forward
        self.window_past   = window_past
        self.window_future = window_future

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

        # 1) Embedding + LSTM phrase
        self.D_m        = D_m
        self.embedding_dim = embedding_dim
        self.embedding  = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.lstm_sen   = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=D_m,
            num_layers=2,
            bidirectional=False,
            dropout=dropout
        ).to(self.device)

        # 2) Modèle séquentiel sur énoncés
        self.base_model = base_model
        if base_model == 'LSTM':
            self.lstm = nn.LSTM(
                input_size=D_m,
                hidden_size=D_e,
                num_layers=2,
                bidirectional=True,
                dropout=dropout
            ).to(self.device)
        elif base_model == 'GRU':
            self.gru = nn.GRU(
                input_size=D_m,
                hidden_size=D_e,
                num_layers=2,
                bidirectional=True,
                dropout=dropout
            ).to(self.device)
        elif base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e).to(self.device)
        else:
            raise NotImplementedError(f"Unknown base_model: {base_model}")

        # 3) Attention d'arêtes et mapping
        n_relations            = 2 * n_speakers ** 2
        self.att_model         = MaskedEdgeAttention(2 * D_e, max_seq_len).to(self.device)
        self.nodal_attention   = nodal_attention

        # 4) Raisonnement GCN
        self.graph_net = GraphNetwork(
            in_feats      = 2 * D_e,
            out_feats     = 7,                # MELD: 7 classes d'émotion
            n_relations   = n_relations,
            max_seq_len   = max_seq_len,
            graph_hid_dim = graph_hidden_size,
            dropout       = dropout,
            no_cuda       = no_cuda
        ).to(self.device)

        # 5) Mapping type d'arête
        edge_type_mapping = {}
        for i in range(n_speakers):
            for j in range(n_speakers):
                edge_type_mapping[f"{i}{j}0"] = len(edge_type_mapping)
                edge_type_mapping[f"{i}{j}1"] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

    def init_pretrained_embeddings(self, pretrained_vectors):
            # 1) Si c'est un numpy.ndarray, on convertit en Tensor
            if isinstance(pretrained_vectors, np.ndarray):
                pretrained_vectors = torch.from_numpy(pretrained_vectors)
            # 2) On déplace sur le bon device (CPU ou GPU)
            pretrained_vectors = pretrained_vectors.to(self.device)
            # 3) On écrase directement les poids existants et on fige
            with torch.no_grad():
                # copie dans weight.data pour ne pas recréer un Parameter
                self.embedding.weight.data.copy_(pretrained_vectors)
                self.embedding.weight.requires_grad = False

    def forward(self, input_seq, qmask, umask, seq_lengths):
        """
        input_seq: LongTensor (B, U, L)
        qmask:     FloatTensor (U, B, n_speakers)
        umask:     FloatTensor (B, U)
        seq_lengths: list[int]

        Returns:
          log_prob, edge_index, edge_norm, edge_type, edge_index_lengths
        """
        B, U, L = input_seq.size()

        # 1) Encodage phrase
        embed = self.embedding(input_seq.to(self.device))           # (B, U, L, embedding_dim)
        embed = embed.permute(2, 0, 1, 3).contiguous().view(L, B * U, self.embedding_dim)
        Ut, _ = self.lstm_sen(embed)                                # (L, B*U, D_m)
        U_last = Ut[-1]                                             # (B*U, D_m)
        U_utt = U_last.view(B, U, self.D_m)                         # (B, U, D_m)

        # 2) Séquence énoncés
        U_in = U_utt.permute(1, 0, 2)                               # (U, B, D_m)
        if self.base_model == 'LSTM':
            emotions, _ = self.lstm(U_in)
        elif self.base_model == 'GRU':
            emotions, _ = self.gru(U_in)
        else:
            emotions = self.base_linear(U_in)

        # 3) Construction du graphe
        features, edge_index, edge_norm, edge_type, ei_len = batch_graphify(
            emotions,
            qmask.to(self.device),
            seq_lengths,
            self.window_past,
            self.window_future,
            self.edge_type_mapping,
            self.att_model,
            self.device
        )

        # 4) GCN + classification
        log_prob = self.graph_net(
            features, edge_index, edge_norm, edge_type,
            seq_lengths, umask.to(self.device)
        )
        return log_prob, edge_index, edge_norm, edge_type, ei_len