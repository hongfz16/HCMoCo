"""
DDP training for Contrastive Learning
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

from options.train_options import TrainOptions
from learning.contrast_trainer import ContrastTrainer
from networks.build_backbone import build_model
from datasets.util import build_contrast_loader
from datasets.util import build_own_contrast_loader
from memory.build_memory import build_mem


def main():
    args = TrainOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        main_worker(0, ngpus_per_node, args)
    else:
        raise NotImplementedError('Currently only DDP training')


def main_worker(gpu, ngpus_per_node, args):

    # initialize trainer and ddp environment
    trainer = ContrastTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build model
    model, model_ema = build_model(args)

    # build dataset
    # train_dataset, train_loader, train_sampler = \
        # build_contrast_loader(args, ngpus_per_node)
    train_dataset, train_loader, train_sampler = \
        build_own_contrast_loader(args, ngpus_per_node)

    # build memory
    contrast = build_mem(args, len(train_dataset))
    contrast.cuda()

    if args.pretrain is not None:
        ckpt = torch.load(args.pretrain, map_location='cpu')
        update_dict = {}
        unmatched_key = []
        converted_dict = {}
        for k, v in ckpt['model'].items():
            converted_dict[k[7:]] = v
        for k, v in model.state_dict().items():
            if k in converted_dict:
                update_dict[k] = converted_dict[k]
            else:
                unmatched_key.append(k)
                update_dict[k] = v
        print("Unmatched Keys: {}".format(', '.join(unmatched_key)))
        model.load_state_dict(update_dict)
        contrast.load_state_dict(ckpt['contrast'])

    # build criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.mem == 'bank+clip' or args.mem == 'bank+pri3d' or args.mem == 'bank+jointspri3d' or args.mem == 'bank+crosssubjoints' or \
       args.mem == 'bank+geojointspri3d':
        criterion = [criterion, [nn.CrossEntropyLoss().cuda(), nn.CrossEntropyLoss().cuda()]]
    elif args.mem == 'bank+clip+geo':
        criterion = [criterion, []]
    elif args.mem == 'bank+clip+humangps':
        criterion = [criterion, []]
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # wrap up models
    model, model_ema, optimizer = trainer.wrap_up(model, model_ema, optimizer)

    # optional step: synchronize memory
    trainer.broadcast_memory(contrast)

    # check and resume a model
    start_epoch = trainer.resume_model(model, model_ema, contrast, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train(epoch, train_loader, model, model_ema,
                             contrast, criterion, optimizer)

        # log to tensorbard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'])

        # save model
        trainer.save(model, model_ema, contrast, optimizer, epoch)


if __name__ == '__main__':
    main()
