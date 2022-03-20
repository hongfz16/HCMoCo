from __future__ import print_function

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .util import AverageMeter, accuracy
from .base_trainer import BaseTrainer

from networks.util import GaussianSmoothing

try:
    from apex import amp, optimizers
except ImportError:
    pass


class ContrastTrainer(BaseTrainer):
    """trainer for contrastive pretraining"""
    def __init__(self, args):
        super(ContrastTrainer, self).__init__(args)
        # self.lambda1 = 0.1
        self.lambda1 = 2.0
        self.lambda2 = 5.0
        self.lambda3 = 1.0
        self.lambda4 = 1.0
        #self.smoothing = GaussianSmoothing(128, 5, 1)

    def logging(self, epoch, logs, lr):
        """ logging to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
        """
        args = self.args
        if args.rank == 0:
            self.logger.log_value('loss', logs[0], epoch)
            self.logger.log_value('acc', logs[1], epoch)
            self.logger.log_value('jig_loss', logs[2], epoch)
            self.logger.log_value('jig_acc', logs[3], epoch)
            self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, model_ema, optimizer):
        """Wrap up models with apex and DDP

        Args:
          model: model
          model_ema: momentum encoder
          optimizer: optimizer
        """
        args = self.args

        model.cuda(args.gpu)
        if isinstance(model_ema, torch.nn.Module):
            model_ema.cuda(args.gpu)

        # to amp model if needed
        if args.amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.opt_level
            )
            if isinstance(model_ema, torch.nn.Module):
                model_ema = amp.initialize(
                    model_ema, opt_level=args.opt_level
                )
        # to distributed data parallel
        model = DDP(model, device_ids=[args.gpu])

        if isinstance(model_ema, torch.nn.Module):
            self.momentum_update(model.module, model_ema, 0)

        return model, model_ema, optimizer

    def broadcast_memory(self, contrast):
        """Synchronize memory buffers

        Args:
          contrast: memory.
        """
        if self.args.modal == 'RGB':
            dist.broadcast(contrast.memory, 0)
        else:
            dist.broadcast(contrast.memory_1, 0)
            dist.broadcast(contrast.memory_2, 0)

    def resume_model(self, model, model_ema, contrast, optimizer):
        """load checkpoint"""
        args = self.args
        start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model'])
                contrast.load_state_dict(checkpoint['contrast'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if isinstance(model_ema, torch.nn.Module):
                    model_ema.load_state_dict(checkpoint['model_ema'])
                if args.amp:
                    amp.load_state_dict(checkpoint['amp'])
                print("=> resume successfully '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        return start_epoch

    def save(self, model, model_ema, contrast, optimizer, epoch):
        """save model to checkpoint"""
        args = self.args
        if args.local_rank == 0:
            # saving the model to each instance
            print('==> Saving...')
            state = {
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if isinstance(model_ema, torch.nn.Module):
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch))
                torch.save(state, save_file)
            # help release GPU memory
            del state

    def train(self, epoch, train_loader, model, model_ema, contrast,
              criterion, optimizer):
        """one epoch training"""
        args = self.args
        model.train()

        time1 = time.time()
        if args.mem == 'moco':
            outs = self._train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer)
        elif args.mem == 'bank':
            outs = self._train_mem_skeleton3d(epoch, train_loader, model, contrast, criterion, optimizer)
        elif args.mem == 'bank+jointspri3d':
            outs = self._train_bank_joints_pri3d_cmc3(epoch, train_loader, model, contrast, criterion[0], criterion[1], optimizer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return outs

    @staticmethod
    def _global_gather(x):
        all_x = [torch.ones_like(x)
                 for _ in range(dist.get_world_size())]
        dist.all_gather(all_x, x, async_op=False)
        return torch.cat(all_x, dim=0)

    def _shuffle_bn(self, x, model_ema):
        """ Shuffle BN implementation

        Args:
          x: input image on each GPU/process
          model_ema: momentum encoder on each GPU/process
        """
        args = self.args
        local_gp = self.local_group
        bsz = x.size(0)

        # gather x locally for each node
        node_x = [torch.ones_like(x)
                  for _ in range(dist.get_world_size(local_gp))]
        dist.all_gather(node_x, x.contiguous(),
                        group=local_gp, async_op=False)
        node_x = torch.cat(node_x, dim=0)

        # shuffle bn
        shuffle_ids = torch.randperm(
            bsz * dist.get_world_size(local_gp)).cuda()
        reverse_ids = torch.argsort(shuffle_ids)
        dist.broadcast(shuffle_ids, 0)
        dist.broadcast(reverse_ids, 0)

        this_ids = shuffle_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        with torch.no_grad():
            this_x = node_x[this_ids]
            if args.jigsaw:
                k = model_ema(this_x, x_jig=None, mode=1)
            else:
                k = model_ema(this_x, mode=1)

        # globally gather k
        all_k = self._global_gather(k)

        # unshuffle bn
        node_id = args.node_rank
        ngpus = args.ngpus_per_node
        node_k = all_k[node_id*ngpus*bsz:(node_id+1)*ngpus*bsz]
        this_ids = reverse_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        k = node_k[this_ids]

        return k, all_k

    @staticmethod
    def _compute_loss_accuracy(logits, target, criterion, use_depth=None, use_rgb=None):
        """
        Args:
          logits: a list of logits, each with a contrastive task
          target: contrastive learning target
          criterion: typically nn.CrossEntropyLoss
        """
        def acc(l, t):
            acc1 = accuracy(l, t)
            return acc1[0]
        if use_rgb is not None:
            assert use_depth is not None
            depth_ind = (use_depth == 1)
            rgb_ind = (use_rgb == 1)
            together_ind = torch.logical_and(depth_ind, rgb_ind)
            if together_ind.sum() == 0:
                losses = [(logit - logit).sum() for logit in logits[:-2]] + [criterion(logit, target) for logit in logits[-2:]]
                accuracies = [np.array([0]) for logit in logits[:-2]] + [acc(logit, target) for logit in logits[-2:]]
                return losses, accuracies
            losses = [criterion(logit[together_ind], target[together_ind]) for logit in logits]
            accuracies = [acc(logit[together_ind], target[together_ind]) for logit in logits]
        elif use_depth is not None:
            depth_ind = (use_depth == 1)
            if use_depth.sum() == 0:
                losses = [(logit - logit).sum() for logit in logits[:-2]] + [criterion(logit, target) for logit in logits[-2:]]
                accuracies = [np.array([0]) for logit in logits[:-2]] + [acc(logit, target) for logit in logits[-2:]]
                return losses, accuracies
            losses = []
            accuracies = []
            for i in range(len(logits)):
                if i <= 3:
                    losses.append(criterion(logits[i][depth_ind], target[depth_ind]))
                    accuracies.append(acc(logits[i][depth_ind], target[depth_ind]))
                else:
                    losses.append(criterion(logits[i], target))
                    accuracies.append(acc(logits[i], target))
        else:
            losses = [criterion(logit, target) for logit in logits]
            accuracies = [acc(logit, target) for logit in logits]

        return losses, accuracies

    def _train_moco(self, epoch, train_loader, model, model_ema, contrast,
                    criterion, optimizer):
        """
        MoCo encoder style training. This needs two forward passes,
        one for normal encoder, and one for moco encoder
        """
        args = self.args
        model.train()
        model_ema.eval()

        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        model_ema.apply(set_bn_train)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        loss_jig_meter = AverageMeter()
        acc_jig_meter = AverageMeter()

        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)

            inputs = data[0].float().cuda(args.gpu, non_blocking=True)
            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # split into two crops
            x1, x2 = torch.split(inputs, [3, 3], dim=1)

            # shuffle BN for momentum encoder
            k, all_k = self._shuffle_bn(x2, model_ema)

            # loss and metrics
            if args.jigsaw:
                inputs_jig = data[2].float().cuda(args.gpu, non_blocking=True)
                bsz, m, c, h, w = inputs_jig.shape
                inputs_jig = inputs_jig.view(bsz * m, c, h, w)
                q, q_jig = model(x1, inputs_jig)
                if args.modal == 'CMC':
                    q1, q2 = torch.chunk(q, 2, dim=1)
                    q1_jig, q2_jig = torch.chunk(q_jig, 2, dim=1)
                    k1, k2 = torch.chunk(k, 2, dim=1)
                    all_k1, all_k2 = torch.chunk(all_k, 2, dim=1)
                    output = contrast(q1, k1, q2, k2, q2_jig, q1_jig,
                                      all_k1, all_k2)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * (losses[0] + losses[1]) + \
                        args.beta * (losses[2] + losses[3])
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = 0.5 * (losses[2] + losses[3])
                    update_acc_jig = 0.5 * (accuracies[2] + accuracies[3])
                else:
                    output = contrast(q, k, q_jig, all_k)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * losses[0] + \
                        args.beta * losses[1]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = losses[1]
                    update_acc_jig = accuracies[1]
            else:
                q = model(x1)
                if args.modal == 'CMC':
                    q1, q2 = torch.chunk(q, 2, dim=1)
                    k1, k2 = torch.chunk(k, 2, dim=1)
                    all_k1, all_k2 = torch.chunk(all_k, 2, dim=1)
                    output = contrast(q1, k1, q2, k2,
                                      all_k1=all_k1,
                                      all_k2=all_k2)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0] + losses[1]
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])
                else:
                    output = contrast(q, k, all_k=all_k)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])

            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update meters
            loss_meter.update(update_loss.item(), bsz)
            loss_jig_meter.update(update_loss_jig.item(), bsz)
            acc_meter.update(update_acc[0], bsz)
            acc_jig_meter.update(update_acc_jig[0], bsz)

            # update momentum encoder
            self.momentum_update(model.module, model_ema, args.alpha)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
                          'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
                          'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
                          'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=loss_meter, acc=acc_meter,
                           loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))
                    sys.stdout.flush()

        return loss_meter.avg, acc_meter.avg, loss_jig_meter.avg, acc_jig_meter.avg

    def _train_mem(self, epoch, train_loader, model, contrast,
                   criterion, optimizer):
        """
        Training based on memory bank mechanism. Only one forward pass.
        """
        args = self.args
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        loss_jig_meter = AverageMeter()
        acc_jig_meter = AverageMeter()

        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            # import pdb; pdb.set_trace()

            inputs = data[0].float().cuda(args.gpu, non_blocking=True)
            index = data[1].cuda(args.gpu, non_blocking=True)
            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # compute feature
            if args.jigsaw:
                inputs_jig = data[2].float().cuda(args.gpu, non_blocking=True)
                bsz, m, c, h, w = inputs_jig.shape
                inputs_jig = inputs_jig.view(bsz * m, c, h, w)
                f, f_jig = model(inputs, inputs_jig)
            else:
                if args.modal == 'DS':
                    s = data[2].cuda(args.gpu, non_blocking=True)
                    f = model(inputs, s)
                else:
                    f = model(inputs)

            # gather all feature and index
            all_f = self._global_gather(f)
            all_index = self._global_gather(index)

            # loss and metrics
            if args.jigsaw:
                if args.modal.startswith('CMC') or \
                   args.modal.startswith('RGBHHA') or \
                   args.modal.startswith('RGBD'):
                    f1, f2 = torch.chunk(f, 2, dim=1)
                    f1_jig, f2_jig = torch.chunk(f_jig, 2, dim=1)
                    all_f1, all_f2 = torch.chunk(all_f, 2, dim=1)
                    output = contrast(f1, f2, index, f2_jig, f1_jig,
                                      all_f1, all_f2, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * (losses[0] + losses[1]) + \
                        args.beta * (losses[2] + losses[3])
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = 0.5 * (losses[2] + losses[3])
                    update_acc_jig = 0.5 * (accuracies[2] + accuracies[3])
                else:
                    output = contrast(f, index, f_jig, all_f, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * losses[0] + \
                        args.beta * losses[1]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = losses[1]
                    update_acc_jig = accuracies[1]
            else:
                if args.modal.startswith('CMC') or \
                   args.modal.startswith('RGBHHA') or \
                   args.modal.startswith('RGBD') or \
                   args.modal.startswith('DS'):
                    f1, f2 = torch.chunk(f, 2, dim=1)
                    all_f1, all_f2 = torch.chunk(all_f, 2, dim=1)
                    output = contrast(f1, f2, index, None, None,
                                      all_f1, all_f2, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0] + losses[1]
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])
                else:
                    output = contrast(f, index, None, all_f, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])

            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update meters
            loss_meter.update(update_loss.item(), bsz)
            loss_jig_meter.update(update_loss_jig.item(), bsz)
            acc_meter.update(update_acc[0], bsz)
            acc_jig_meter.update(update_acc_jig[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
                          'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
                          'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
                          'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=loss_meter, acc=acc_meter,
                           loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))
                    sys.stdout.flush()

        return loss_meter.avg, acc_meter.avg, loss_jig_meter.avg, acc_jig_meter.avg
    
    def _train_mem_skeleton3d(self, epoch, train_loader, model, contrast, criterion, optimizer):
        """
        Training based on memory bank mechanism. Only one forward pass.
        """
        args = self.args
        assert not args.jigsaw
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss1_meter = AverageMeter()
        acc1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        acc2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        acc3_meter = AverageMeter()

        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)

            inputs = data[0].float().cuda(args.gpu, non_blocking=True)
            index = data[1].cuda(args.gpu, non_blocking=True)
            skeleton = data[2].cuda(args.gpu, non_blocking=True)
            depth_mask = data[7].cuda(args.gpu, non_blocking=True)
            if args.arch == 'HRNetPN':
                grid_xy = data[12].cuda(args.gpu, non_blocking=True)
                original_h = data[13][0].item()
                original_w = data[14][0].item()
                mean = data[15].cuda(args.gpu, non_blocking=True)
            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # compute feature
            if args.jigsaw:
                raise NotImplementedError
            else:
                if args.arch == 'HRNetPN':
                    f = model(inputs, skeleton, depth_mask, grid_xy, original_h, original_w, mean)
                else:
                    f = model(inputs, skeleton)

            # gather all feature and index
            all_f = self._global_gather(f)
            all_index = self._global_gather(index)

            # loss and metrics
            f1, f2, f3 = torch.chunk(f, 3, dim=1)
            all_f1, all_f2, all_f3 = torch.chunk(all_f, 3, dim=1)
            output = contrast(f1, f2, f3, index, all_f1, all_f2, all_f3, all_index)
            if args.modality_missing:
                use_depth = data[6].cuda(args.gpu, non_blocking=True)
            else:
                use_depth = None
            try:
                use_rgb = data[11].cuda(args.gpu, non_blocking=True)
            except:
                use_rgb = None
            losses, accuracies = self._compute_loss_accuracy(
                logits=output[:-1], target=output[-1],
                criterion=criterion, use_depth=use_depth, use_rgb=use_rgb)
            loss = sum(losses)
            update_loss_12 = 0.5 * (losses[0] + losses[1])
            update_acc_12 = 0.5 * (accuracies[0] + accuracies[1])
            update_loss_23 = 0.5 * (losses[2] + losses[3])
            update_acc_23 = 0.5 * (accuracies[2] + accuracies[3])
            update_loss_13 = 0.5 * (losses[4] + losses[5])
            update_acc_13 = 0.5 * (accuracies[4] + accuracies[5])
                
            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update meters
            loss1_meter.update(update_loss_12.item(), bsz)
            acc1_meter.update(update_acc_12[0], bsz)
            loss2_meter.update(update_loss_23.item(), bsz)
            acc2_meter.update(update_acc_23[0], bsz)
            loss3_meter.update(update_loss_13.item(), bsz)
            acc3_meter.update(update_acc_13[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'l_I {loss1.val:.3f} ({loss1.avg:.3f})\t'
                          'a_I {acc1.val:.3f} ({acc1.avg:.3f})\t'
                          'l_I {loss2.val:.3f} ({loss2.avg:.3f})\t'
                          'a_I {acc2.val:.3f} ({acc2.avg:.3f})\t'
                          'l_I {loss3.val:.3f} ({loss3.avg:.3f})\t'
                          'a_I {acc3.val:.3f} ({acc3.avg:.3f})'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time,
                           loss1=loss1_meter, acc1=acc1_meter, loss2=loss2_meter, acc2=acc2_meter,
                           loss3=loss3_meter, acc3=acc3_meter))
                    sys.stdout.flush()

        return loss1_meter.avg, acc1_meter.avg, loss2_meter.avg, acc2_meter.avg, loss3_meter.avg, acc3_meter.avg

    def _compute_soft_pri3d_loss_accuracy(self, _feat1_, _feat2_, _depth, criterion_pri3d, use_depth=None, depth_mask=None, scale=None):
        _feat1 = _feat1_
        _feat2 = _feat2_
        depth = _depth
        def merge_all_res(x):
            ALIGN_CORNERS=False
            x0_h, x0_w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x = torch.cat([x[0], x1, x2, x3], 1)
            return x
        if isinstance(_feat1, list) and isinstance(_feat2, list):
            merge1 = merge_all_res(_feat1)
            merge2 = merge_all_res(_feat2)
            h, w = _feat1[0].size(2), _feat1[0].size(3)
        else:
            merge1 = _feat1
            merge2 = _feat2
            h, w = merge1.size(2), merge2.size(3)

        if use_depth is not None:
            if use_depth.sum() == 0:
                return [(merge1 - merge1 + merge2 - merge2).mean() for _ in range(2)], [np.array([0]), np.array([0])]

        bs, fdim = merge1.size(0), merge1.size(1)
        resized_depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='nearest')
        resized_depth = resized_depth.reshape(bs, h * w)
        merge1 = merge1.reshape(bs, fdim, h * w)
        merge2 = merge2.reshape(bs, fdim, h * w)

        # valid_depth_prob = (resized_depth > 0).float()
        valid_depth_prob = F.interpolate(depth_mask.unsqueeze(1).float(), size=(h,w), mode='nearest')
        valid_depth_prob = valid_depth_prob.reshape(bs, h * w)

        valid_depth_prob_sum = valid_depth_prob.sum(-1)
        mask = valid_depth_prob_sum > 0
        valid_depth_prob = valid_depth_prob[mask]
        merge1 = merge1[mask]
        merge2 = merge2[mask]
        bs = mask.sum()

        num_samples = self.args.pri3d_num_samples_per_image
        random_sample_ind = valid_depth_prob.multinomial(num_samples=num_samples, replacement=True) # bs, num_samples
        random_sample_ind_copy = random_sample_ind.clone()
        random_sample_ind = random_sample_ind.unsqueeze(1).repeat(1, fdim, 1)

        sampled_merge1 = torch.gather(merge1, 2, random_sample_ind)
        sampled_merge2 = torch.gather(merge2, 2, random_sample_ind)

        sampled_merge1 = torch.nn.functional.normalize(sampled_merge1, dim=1)
        sampled_merge2 = torch.nn.functional.normalize(sampled_merge2, dim=1)

        rgb2depth_logits = torch.matmul(sampled_merge2.permute(0, 2, 1), sampled_merge1)
        depth2rgb_logits = torch.matmul(sampled_merge1.permute(0, 2, 1), sampled_merge2)

        rgb2depth_logits = rgb2depth_logits / self.args.temperature
        depth2rgb_logits = depth2rgb_logits / self.args.temperature

        ### START: generate soft target
        sample_xy = torch.stack([random_sample_ind_copy // w, random_sample_ind_copy % w], -1).float()
        sample_xy_dist = sample_xy.reshape(bs, num_samples, 1, 2) - sample_xy.reshape(bs, 1, num_samples, 2)
        sample_xy_dist = torch.sqrt((sample_xy_dist ** 2).sum(-1))

        soft_target = torch.softmax(-sample_xy_dist, 1)
        rgb2depth_logsoft = torch.nn.functional.log_softmax(rgb2depth_logits, dim=1)
        depth2rgb_logsoft = torch.nn.functional.log_softmax(depth2rgb_logits, dim=1)
        
        losses = [
            -(soft_target * rgb2depth_logsoft).sum(-2).mean(),
            -(soft_target * depth2rgb_logsoft).sum(-2).mean()
        ]
        ### END
        target = torch.arange(num_samples).long().cuda().unsqueeze(0).repeat(bs, 1)
        
        rgb2depth_pred = rgb2depth_logits.argmax(-2)
        depth2rgb_pred = depth2rgb_logits.argmax(-2)
        rgb2depth_acc = (rgb2depth_pred == target).sum(-1).float() / num_samples
        depth2rgb_acc = (depth2rgb_pred == target).sum(-1).float() / num_samples
        acces = [rgb2depth_acc.float().mean(), depth2rgb_acc.float().mean()]

        return losses, acces

    def _gaussian_joint_pooling(self, feat_map, original_joints2d):
        pad_feat_map = F.pad(feat_map, (2, 2, 2, 2), mode='reflect')
        smoothing = GaussianSmoothing(128, 5, 1).cuda()
        smooth_feat_map = smoothing(pad_feat_map)

        assert smooth_feat_map.shape == feat_map.shape
        bs, fdim, h, w = feat_map.shape

        downsampled_joints2d = original_joints2d // 4
        downsampled_joints2d = downsampled_joints2d.long()
        downsampled_joints2d[downsampled_joints2d >= h] = h - 1
        downsampled_joints2d[downsampled_joints2d < 0] = 0
        sample_ind = downsampled_joints2d[:, :, 0] * h + downsampled_joints2d[:, :, 1]
        assert sample_ind.max() < h * w and sample_ind.min() >= 0
        sample_ind = sample_ind.unsqueeze(1).repeat(1, fdim, 1)
        joints_feat = torch.gather(smooth_feat_map.reshape(bs, fdim, h*w), 2, sample_ind)

        return joints_feat

    def _compute_joints_pri3d_loss_accuracy(self, _feat_rgb, _feat_d, _feat_joints2d, criterion_pri3d, original_joints2d, joints_vis, use_depth=None, ready_joint_feat=False, use_rgb2d = False):
        # if use_depth is not None:
        #     if use_depth.sum() == 0:
        #         return [(_feat_rgb - _feat_rgb + _feat_d - _feat_d).mean() for _ in range(2)], [np.array([0]), np.array([0])]

        if not ready_joint_feat:
            bs, fdim, h, w = _feat_rgb.shape
            assert h == w

        if not ready_joint_feat:
            downsampled_joints2d = original_joints2d // 4
            downsampled_joints2d = downsampled_joints2d.long()
            downsampled_joints2d[downsampled_joints2d >= h] = h - 1
            downsampled_joints2d[downsampled_joints2d < 0] = 0
            sample_ind = downsampled_joints2d[:, :, 0] * h + downsampled_joints2d[:, :, 1]
            assert sample_ind.max() < h * w and sample_ind.min() >= 0
            sample_ind = sample_ind.unsqueeze(1).repeat(1, fdim, 1)
            rgb_joints_feat = torch.gather(_feat_rgb.reshape(bs, fdim, h*w), 2, sample_ind)
            d_joints_feat = torch.gather(_feat_d.reshape(bs, fdim, h*w), 2, sample_ind)
        else:
            rgb_joints_feat = _feat_rgb.permute(0, 2, 1)
            d_joints_feat = _feat_d.permute(0, 2, 1)
            
        rgb_joints_feat = torch.nn.functional.normalize(rgb_joints_feat, dim=1)
        d_joints_feat = torch.nn.functional.normalize(d_joints_feat, dim=1)
        joints2d_feat = torch.nn.functional.normalize(_feat_joints2d, dim=-1)

        rgb2joints_logits = torch.matmul(joints2d_feat, rgb_joints_feat)
        d2joints_logits = torch.matmul(joints2d_feat, d_joints_feat)

        if use_rgb2d:
            rgb2d_logits = torch.matmul(rgb_joints_feat.permute(0, 2, 1), d_joints_feat)

        rgb2joints_logits = rgb2joints_logits / self.args.temperature
        d2joints_logits = d2joints_logits / self.args.temperature

        if use_rgb2d:
            rgb2d_logits = rgb2d_logits / self.args.temperature

        bs = original_joints2d.shape[0]
        target = torch.arange(joints_vis.shape[1]).long().unsqueeze(0).repeat(bs, 1).cuda()
        assert joints_vis.shape[0] == target.shape[0]
        assert joints_vis.shape[1] == target.shape[1]
        target[torch.logical_not(joints_vis)] = -100

        depth_target = target.clone()
        if use_depth is not None:
            depth_target[torch.logical_not(use_depth)] = -100

        losses = [
            criterion_pri3d[0](rgb2joints_logits, target),
            criterion_pri3d[1](d2joints_logits, depth_target),
        ]

        if use_rgb2d:
            losses.append(criterion_pri3d[1](rgb2d_logits, depth_target))

        rgb2joints_pred = rgb2joints_logits.argmax(-2)
        d2joints_pred = d2joints_logits.argmax(-2)
        
        target_denom = (target != -100).sum(-1)
        target_denom_copy = (target != -100).sum(-1)
        target_denom[target_denom == 0] = 1

        depth_target_denom = (depth_target != -100).sum(-1)
        depth_target_denom_copy = (depth_target != -100).sum(-1)
        depth_target_denom[depth_target_denom == 0] = 1

        rgb2joints_acc = (rgb2joints_pred == target).sum(-1).float() / target_denom
        d2joints_acc = (d2joints_pred == depth_target).sum(-1).float() / depth_target_denom

        # rgb2joints_acc[target_denom_copy == 0] = 0.5
        # d2joints_acc[depth_target_denom_copy == 0] = 0.5
        rgb2joints_acc_valid = rgb2joints_acc[target_denom_copy != 0]
        d2joints_acc_valid = d2joints_acc[depth_target_denom_copy != 0]

        acces = [rgb2joints_acc_valid.float().mean(), d2joints_acc_valid.float().mean()]

        if use_rgb2d:
            rgb2d_pred = rgb2d_logits.argmax(-2)
            rgb2d_acc = (rgb2d_pred == depth_target).sum(-1).float() / depth_target_denom
            rgb2d_acc_valid = rgb2d_acc[depth_target_denom_copy != 0]
            acces.append(rgb2d_acc_valid.float().mean())

        return losses, acces

    def _compute_cross_subject_joints_pri3d_loss(self, _feat_rgb, _feat_d, _feat_joints2d, criterion_pri3d, original_joints2d, joints_vis, use_depth=None, ready_joint_feat=False, index=None, memory=None, use_rgb=None):
        def merge_all_res(x):
            ALIGN_CORNERS=False
            x0_h, x0_w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x = torch.cat([x[0], x1, x2, x3], 1)
            return x

        if not ready_joint_feat:
            if isinstance(_feat_rgb, list) and isinstance(_feat_d, list):
                _feat_rgb = merge_all_res(_feat_rgb)
                _feat_d = merge_all_res(_feat_d)
            
            if use_depth is not None:
                if use_depth.sum() == 0:
                    return [(_feat_rgb - _feat_rgb + _feat_d - _feat_d).mean() for _ in range(4)], [np.array([0]) for _ in range(4)]

            bs, fdim, h, w = _feat_rgb.shape
            num_joints = original_joints2d.shape[-2]
            assert h == w
            downsampled_joints2d = original_joints2d // 4
            downsampled_joints2d = downsampled_joints2d.long()
            downsampled_joints2d[downsampled_joints2d >= h] = h - 1
            downsampled_joints2d[downsampled_joints2d < 0] = 0
            sample_ind = downsampled_joints2d[:, :, 0] * h + downsampled_joints2d[:, :, 1]
            assert sample_ind.max() < h * w and sample_ind.min() >= 0
            sample_ind = sample_ind.unsqueeze(1).repeat(1, fdim, 1)
            rgb_joints_feat = torch.gather(_feat_rgb.reshape(bs, fdim, h*w), 2, sample_ind)
            d_joints_feat = torch.gather(_feat_d.reshape(bs, fdim, h*w), 2, sample_ind)
        else:
            bs, num_joints, fdim = _feat_rgb.shape
            rgb_joints_feat = _feat_rgb.permute(0, 2, 1)
            d_joints_feat = _feat_d.permute(0, 2, 1)

        temp = self.args.temperature
        rgb_joints_feat = torch.nn.functional.normalize(rgb_joints_feat, dim=1).permute(0, 2, 1).reshape(bs * num_joints, fdim)
        d_joints_feat = torch.nn.functional.normalize(d_joints_feat, dim=1).permute(0, 2, 1).reshape(bs * num_joints, fdim)

        cat_feat = torch.cat([rgb_joints_feat, d_joints_feat], 0)
        logits = torch.matmul(cat_feat, cat_feat.permute(1, 0)) / temp # 2*num_joints*bs x 2*num_joints*bs
        logsoftmax_logits = torch.nn.functional.log_softmax(logits, 1)
        positive_ind = torch.zeros([num_joints, 2 * bs * num_joints], dtype=torch.int32).cuda()
        for i in range(num_joints):
            positive_ind[i, i::num_joints] = 1
        positive_ind = positive_ind.repeat(2*bs, 1).reshape(2*bs*num_joints, 2*bs*num_joints)
        ind = np.diag_indices(2 * bs * num_joints)
        positive_ind[ind[0], ind[1]] = 0
        not_use_depth = torch.logical_not(use_depth)
        not_use_depth = not_use_depth.reshape(bs, 1).repeat(1, num_joints).reshape(bs * num_joints)
        not_use_rgb = torch.logical_not(use_rgb)
        not_use_rgb = not_use_rgb.reshape(bs, 1).repeat(1, num_joints).reshape(bs * num_joints)
        not_use_depth = torch.cat([not_use_rgb, not_use_depth])
        positive_ind[not_use_depth, :] = 0
        positive_ind[:, not_use_depth] = 0

        positive_logits = logsoftmax_logits * positive_ind
        positive_ind_sum = positive_ind.sum(-1)
        positive_ind_sum[positive_ind_sum == 0] = 1
        positive_logits_mean = -positive_logits.sum(-1) / positive_ind_sum
        loss = positive_logits_mean.mean()
        return [loss], [np.array([0])]

    def _train_bank_joints_pri3d_cmc3(self, epoch, train_loader, model, contrast,
                                      criterion_contrast, criterion_pri3d, optimizer):
        args = self.args
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc2_meter = AverageMeter()
        acc3_meter = AverageMeter()

        loss_jig_meter = AverageMeter()
        acc_jig_meter = AverageMeter()
        loss_rgb2depth_meter = AverageMeter()
        loss_depth2rgb_meter = AverageMeter()
        acc_rgb2depth_meter = AverageMeter()
        acc_depth2rgb_meter = AverageMeter()

        loss_rgb2joint_meter = AverageMeter()
        loss_d2joint_meter = AverageMeter()
        acc_rgb2joint_meter = AverageMeter()
        acc_d2joint_meter = AverageMeter()

        loss_scl_meter = AverageMeter()

        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)

            inputs = data[0].float().cuda(args.gpu, non_blocking=True)
            index = data[1].cuda(args.gpu, non_blocking=True)
            skeleton = data[2].cuda(args.gpu, non_blocking=True)
            original_joints2d = data[4].cuda(args.gpu, non_blocking=True)
            joints_vis = data[5].cuda(args.gpu, non_blocking=True)
            depth_mask = data[7].cuda(args.gpu, non_blocking=True)
            scale = data[8].cuda(args.gpu, non_blocking=True)
            if args.arch == 'HRNetPN':
                grid_xy = data[12].cuda(args.gpu, non_blocking=True)
                original_h = data[13][0].item()
                original_w = data[14][0].item()
                mean = data[15].cuda(args.gpu, non_blocking=True)
            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # compute feature
            if args.arch == 'HRNetPN':
                _feat1, _feat2, _feat3, f, aux_dict = model(inputs, skeleton, depth_mask, grid_xy, original_h, original_w, mean, return_fm = True)
            else:
                _feat1, _feat2, _feat3, f, aux_dict = model(inputs, skeleton, return_fm = True)

            # gather all feature and index
            all_f = self._global_gather(f)
            all_index = self._global_gather(index)

            # loss and metrics
            f1, f2, f3 = torch.chunk(f, 3, dim=1)
            all_f1, all_f2, all_f3 = torch.chunk(all_f, 3, dim=1)
            output = contrast(f1, f2, f3, index, all_f1, all_f2, all_f3, all_index)
            if args.modality_missing:
                use_depth = data[6].cuda(args.gpu, non_blocking=True)
            else:
                use_depth = None
            try:
                use_rgb = data[11].cuda(args.gpu, non_blocking=True)
            except:
                use_rgb = None
            losses, accuracies = self._compute_loss_accuracy(
                logits=output[:-1], target=output[-1],
                criterion=criterion_contrast, use_depth=use_depth)

            losses_clip, accuracies_clip = self._compute_soft_pri3d_loss_accuracy(
                aux_dict['linear_merge1'], aux_dict['linear_merge2'], inputs[:, 3, :, :], criterion_pri3d, use_depth=use_depth, depth_mask=depth_mask, scale=scale)
            
            loss_joints, accuracies_joints = self._compute_joints_pri3d_loss_accuracy(
                aux_dict['linear_merge1'], aux_dict['linear_merge2'], _feat3, criterion_pri3d, original_joints2d, joints_vis, use_depth=use_depth
            )

            loss_scl, _ = self._compute_cross_subject_joints_pri3d_loss(
                aux_dict['linear_merge1'], aux_dict['linear_merge2'], None, criterion_pri3d, original_joints2d, joints_vis, use_depth=use_depth, index=index, memory=contrast.memory_3, use_rgb=use_rgb,
            )

            loss = sum(losses) + sum(losses_clip) + sum(loss_joints) + sum(loss_scl)

            update_loss_12 = 0.5 * (losses[0] + losses[1])
            update_acc_12 = 0.5 * (accuracies[0] + accuracies[1])
            update_loss_23 = 0.5 * (losses[2] + losses[3])
            update_acc_23 = 0.5 * (accuracies[2] + accuracies[3])
            update_loss_13 = 0.5 * (losses[4] + losses[5])
            update_acc_13 = 0.5 * (accuracies[4] + accuracies[5])

            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update meters
            loss_meter.update(loss.item(), bsz)
            acc1_meter.update(update_acc_12.item(), bsz)
            acc2_meter.update(update_acc_23.item(), bsz)
            acc3_meter.update(update_acc_13.item(), bsz)

            loss_rgb2depth_meter.update(losses_clip[0].item(), bsz)
            loss_depth2rgb_meter.update(losses_clip[1].item(), bsz)
            acc_rgb2depth_meter.update(accuracies_clip[0].item(), bsz)
            acc_depth2rgb_meter.update(accuracies_clip[1].item(), bsz)

            loss_rgb2joint_meter.update(loss_joints[0].item(), bsz)
            loss_d2joint_meter.update(loss_joints[1].item(), bsz)
            acc_rgb2joint_meter.update(accuracies_joints[0].item(), bsz)
            acc_d2joint_meter.update(accuracies_joints[1].item(), bsz)

            loss_scl_meter.update(loss_scl[0].item(), bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'L {loss.val:.3f} ({loss.avg:.3f})\t'
                          'a_I {acc1.avg:.3f} {acc2.avg:.3f} {acc3.avg:.3f}\t'
                          'p3d {loss_rgb2depth.avg:.3f} {acc_rgb2depth.avg:.3f} {loss_depth2rgb.avg:.3f} {acc_depth2rgb.avg:.3f}\t'
                          'j {loss_rgb2joint.avg:.3f} {acc_rgb2joint.avg:.3f} {loss_d2joint.avg:.3f} {acc_d2joint.avg:.3f}\t'
                          'scl {loss_scl_meter.avg:.3f}'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=loss_meter,
                           data_time=data_time, acc1=acc1_meter, acc2=acc2_meter, acc3=acc3_meter,
                           loss_rgb2depth=loss_rgb2depth_meter, loss_depth2rgb=loss_depth2rgb_meter,
                           acc_rgb2depth=acc_rgb2depth_meter, acc_depth2rgb=acc_depth2rgb_meter,
                           loss_rgb2joint=loss_rgb2joint_meter, acc_rgb2joint=acc_rgb2joint_meter,
                           loss_d2joint=loss_d2joint_meter, acc_d2joint=acc_d2joint_meter,
                           loss_scl_meter=loss_scl_meter))
                    sys.stdout.flush()

        return loss_meter.avg, acc1_meter.avg, loss_jig_meter.avg, acc_jig_meter.avg

    @staticmethod
    def momentum_update(model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(1 - m, p1.detach().data)
