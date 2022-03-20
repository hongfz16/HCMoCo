"""
DDP training for Linear Probing
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np

from options.train_options import TrainOptions
from learning.segment_trainer import SegTrainer
from networks.build_backbone import build_model
from networks.build_linear import build_segmentor
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
    trainer = SegTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build encoder and classifier
    model, _ = build_model(args)
    classifier = build_segmentor(args)

    # build dataset
    train_loader, train_dataset, val_loader, train_sampler = \
        build_own_contrast_loader(args, ngpus_per_node, need_gt=False, need_val=True)

    contrast = build_mem(args, len(train_dataset))
    contrast.cuda()

    if args.pretrain is not None and not args.resume:
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

    weights = [1.448,49.234,49.483,48.030,49.247,49.492,48.018,49.704,50.052,49.369,49.694,50.090,49.425,49.459,45.846,47.156,45.868,47.197,44.167,42.789,44.341,48.632,48.873,48.644,49.004]
    class_weights = torch.from_numpy(np.array(weights).astype(np.float32))
    
    criterion_seg = [nn.CrossEntropyLoss(ignore_index=255, weight=class_weights).cuda()]
    optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # load pre-trained ckpt for encoder
    # model = trainer.load_encoder_weights(model)

    # wrap up models
    model, classifier = trainer.wrap_up(model, classifier)

    # check and resume a classifier
    start_epoch = trainer.resume_model(model, contrast, classifier, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    best_miou = -1

    # routine
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train_soft_joint_pri3d(epoch, train_loader, model, classifier, contrast,
                             criterion[0], criterion[1], criterion_seg, optimizer)

        # log to tensorbard
        # trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'], train=True)

        # evaluation and logging
        # if args.rank % ngpus_per_node == 0:
        # if epoch % args.save_freq == 0:
        res = trainer.validate(epoch, val_loader, model,
                                classifier, criterion_seg)
            # trainer.logging(epoch, outs, train=False)
        if args.test_type == 0:
            ref_miou = res['rgbd']['miou']
        elif args.test_type == 1:
            ref_miou = res['rgb']['miou']
        elif args.test_type == 2:
            ref_miou = res['d']['miou']
        
        if ref_miou > best_miou:
            best_miou = ref_miou
            trainer.save_seg_models(model, classifier, epoch, res)

        # saving model
        trainer.save(model, contrast, classifier, optimizer, epoch)


if __name__ == '__main__':
    main()
