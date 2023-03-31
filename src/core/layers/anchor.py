import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda, create_mask
from ..utils.constants import VERY_SMALL_NUMBER, INF

'''
这段代码定义了AnchorGCN模型，是一种基于图卷积神经网络（GCN）的模型。模型主要包含了AnchorGCNLayer和AnchorGCN两个类。

AnchorGCNLayer类实现了一层AnchorGCN，用于对输入特征进行线性变换和图卷积操作。其中，线性变换使用torch.matmul实现，图卷积操作则可以选择使用邻接矩阵进行传播，也可以使用节点-锚点邻接矩阵进行传播。在进行节点-锚点邻接矩阵传播时，先将节点特征进行标准化，然后通过锚点-节点邻接矩阵和节点特征进行乘积，再通过节点-锚点邻接矩阵进行传播。在传播过程中，还可以选择是否加入Batch Normalization操作。

AnchorGCN类实现了整个AnchorGCN模型，包含了多层AnchorGCNLayer。在forward函数中，对输入特征进行多层AnchorGCNLayer的处理，最后输出分类结果。
'''

'''
这个函数的作用是从节点向量 node_vec 中随机采样 s 个锚点（anchors），并返回采样到的锚点向量和其在 node_vec 中的索引。
具体来说，node_vec 的大小为 (n, f)，其中 n 表示节点数量，f 表示每个节点的特征维度。函数返回的锚点向量大小为 (s, f)。'''
def sample_anchors(node_vec, s):
    idx = torch.randperm(node_vec.size(0))[:s]
    return node_vec[idx], idx

def batch_sample_anchors(node_vec, ratio, node_mask=None, device=None):
    '''
    node_vec: 节点特征矩阵，大小为 (batch_size, num_nodes, feature_dim)。
    ratio: 每个节点采样的锚点数与节点数的比例。
    node_mask: 表示哪些节点需要被采样为锚点，如果为 None，则表示所有节点都需要采样为锚点。
    device: 运行的设备。
    返回值:
    anchor_vec: 采样得到的锚点矩阵，大小为 (batch_size, num_anchors, feature_dim)。
    anchor_mask: 表示哪些位置是真实的锚点，哪些位置是填充用的，大小为 (batch_size, max_num_anchors)。
    idx: 采样得到的锚点在节点特征矩阵中的下标。
    max_num_anchors: 采样得到的所有锚点中最多的数量。
    '''

    '''
    具体实现如下：

首先对每个节点采样一定数量的锚点，采样数量为 ratio * num_nodes，其中 num_nodes 是该节点的邻居节点数目。
然后根据实际采样到的锚点数量，使用 create_mask 函数生成掩码 anchor_mask，大小为 (batch_size, max_num_anchors)，其中掩码为 1 的位置表示该位置是真实的锚点，0 表示该位置是填充用的。
然后根据采样到的锚点在节点特征矩阵中的下标 idx，从节点特征矩阵中选取对应的特征向量，拼接成一个大小为 (batch_size, max_num_anchors, feature_dim) 的矩阵 anchor_vec，其中填充用的位置使用 0 填充。'''
    idx = []
    num_anchors = []
    max_num_anchors = 0
    for i in range(node_vec.size(0)):
        tmp_num_nodes = int(node_mask[i].sum().item())
        tmp_num_anchors = int(ratio * tmp_num_nodes)
        g_idx = torch.randperm(tmp_num_nodes)[:tmp_num_anchors]
        idx.append(g_idx)
        num_anchors.append(len(g_idx))

        if max_num_anchors < len(g_idx):
            max_num_anchors = len(g_idx)

    anchor_vec = batch_select_from_tensor(node_vec, idx, max_num_anchors, device)
    anchor_mask = create_mask(num_anchors, max_num_anchors, device)

    return anchor_vec, anchor_mask, idx, max_num_anchors

def batch_select_from_tensor(node_vec, idx, max_num_anchors, device=None):
    anchor_vec = []
    for i in range(node_vec.size(0)):
        tmp_anchor_vec = node_vec[i][idx[i]]
        if len(tmp_anchor_vec) < max_num_anchors:
            dummy_anchor_vec = to_cuda(torch.zeros((max_num_anchors - len(tmp_anchor_vec), node_vec.size(-1))), device)
            tmp_anchor_vec = torch.cat([tmp_anchor_vec, dummy_anchor_vec], dim=-2)
        anchor_vec.append(tmp_anchor_vec)

    anchor_vec = torch.stack(anchor_vec, 0)

    return anchor_vec

def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    '''Can be more memory-efficient'''
    anchor_node_adj = node_anchor_adj.transpose(-1, -2)
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=VERY_SMALL_NUMBER) ** -1
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj)

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj



class AnchorGCNLayer(nn.Module):
    """
    Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True, batch_norm=True):
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))

        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(AnchorGCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(AnchorGCNLayer(nhid, nclass, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)

        return x
