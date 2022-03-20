from .mem_bank import RGBMem, CMCMem, CMCMem3
from .mem_moco import RGBMoCo, CMCMoCo


def build_mem(opt, n_data):
    if opt.mem.startswith('bank'):
        mem_func = CMCMem3 
        memory = mem_func(opt.feat_dim, n_data,
                        opt.nce_k, opt.nce_t, opt.nce_m)
    elif opt.mem == 'moco':
        mem_func = RGBMoCo if opt.modal == 'RGB' else CMCMoCo
        memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)
    else:
        raise NotImplementedError(
            'mem not suported: {}'.format(opt.mem))

    return memory
