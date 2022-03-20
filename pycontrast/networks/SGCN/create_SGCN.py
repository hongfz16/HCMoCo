from .graph_utils import adj_mx_from_skeleton
from .sem_gcn import SemGCN, _GraphConv
from .sem_graph_conv import SemGraphConv
from .skeleton_meta import mpii_skeleton, coco_reduce_skeleton

def create_sgcn(name, hidden_dim, num_layers):
    if name == 'mpii':
        adj = adj_mx_from_skeleton(mpii_skeleton)
    elif name == 'coco_reduce':
        adj = adj_mx_from_skeleton(coco_reduce_skeleton)
    else:
        raise NotImplementedError
    model = SemGCN(adj, hidden_dim, coords_dim = (2, hidden_dim), num_layers = num_layers, p_dropout = 0, nodes_group = None)
    return model

def create_gcn_mapper(name, input_dim, output_dim):
    if name == 'mpii':
        adj = adj_mx_from_skeleton(mpii_skeleton)
    elif name == 'coco_reduce':
        adj = adj_mx_from_skeleton(coco_reduce_skeleton)
    else:
        raise NotImplementedError
    mapper = SemGraphConv(input_dim, output_dim, adj)
    return mapper
