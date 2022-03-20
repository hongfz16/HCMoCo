import torch.nn as nn
from .fcn import FCNHead

def build_segmentor(opt):
    n_class = opt.n_class
    # channels = [18, 36, 72, 144]
    channels = [128]
    classifier = FCNHead(
        sum(channels),
        sum(channels),
        n_class,
        num_convs=1,
        kernel_size=1
    )
    return classifier

def build_linear(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    else:
        n_feat = 2048

    classifier = nn.Linear(n_feat, n_class)
    return classifier
