from __future__ import print_function

import os
import sys
import time
import torch
import pickle
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

from .util import AverageMeter, accuracy
from .base_trainer import BaseTrainer

# classes = [
#     'head','hips','leftArm','leftFoot','leftForeArm','leftHand',
#     'leftHandIndex1','leftLeg','leftShoulder','leftToeBase',
#     'leftUpLeg','neck','rightArm','rightFoot','rightForeArm','rightHand',
#     'rightHandIndex1','rightLeg','rightShoulder','rightToeBase','rightUpLeg',
#     'spine','spine1','spine2','Skirt','Dress','Jumpsuit','Top','Trousers','Tshirt',
#     'BG'
# ]

classes = [
    'background',
    'right hip',
    'right knee',
    'right foot',
    'left hip',
    'left knee',
    'left foot',
    'left shoulder',
    'left elbow',
    'left hand',
    'right shoulder',
    'right elbow',
    'right hand',
    'crotch',
    'right thigh',
    'right calf',
    'left thigh',
    'left calf',
    'lower spine',
    'upper spine',
    'head',
    'left arm',
    'left forearm',
    'right arm',
    'right forearm',
]

class SegTrainer(BaseTrainer):
    """trainer for Linear evaluation"""
    def __init__(self, args):
        super(SegTrainer, self).__init__(args)
        # self.lambda1 = 0.5
        # self.lambda2 = 0.05
        # self.lambda1 = 2.0
        # self.lambda2 = 20.0
        self.lambda1 = 0.125
        self.lambda2 = 10.0
        self.f = open(os.path.join(self.args.tb_folder, 'log.txt'), 'a')

    def mprint(self, string):
        self.f.write(string + '\n')
        print(string)
        self.f.flush()

    def logging(self, epoch, logs, lr=None, train=True):
        """ d to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
          train: True of False
        """
        args = self.args
        if args.rank == 0:
            pre = 'train_' if train else 'test_'
            self.logger.log_value(pre+'miou', logs[0], epoch)
            self.logger.log_value(pre+'macc', logs[1], epoch)
            self.logger.log_value(pre+'aacc', logs[2], epoch)
            if train and (lr is not None):
                self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, classifier):
        """Wrap up models with DDP

        Args:
          model: pretrained encoder
          classifier: linear classifier
        """
        args = self.args
        model = model.cuda()
        classifier = classifier.cuda()
        model = DDP(model, device_ids=[args.gpu])
        classifier = DDP(classifier, device_ids=[args.gpu])

        return model, classifier

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

    def gather_eval_counter(self, intersect, union, pred_label, label):
        folder = os.path.join(self.args.tb_folder, 'tmp')
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        dist.barrier()
        pickle.dump({
            'intersect': intersect,
            'union': union,
            'pred_label': pred_label,
            'label': label,
        }, open(os.path.join(folder, 'evaluator_part{}.pkl'.format(self.args.rank)), 'wb'))
        dist.barrier()
        if self.args.rank != 0:
            return intersect, union, pred_label, label
        for i in range(1, self.args.world_size):
            part_file = os.path.join(folder, 'evaluator_part{}.pkl'.format(i))
            part_dict = pickle.load(open(part_file, 'rb'))
            intersect += part_dict['intersect']
            union += part_dict['union']
            pred_label += part_dict['pred_label']
            label += part_dict['label']

        return intersect, union, pred_label, label

    def load_encoder_weights(self, model, contrast):
        """load pre-trained weights for encoder

        Args:
          model: pretrained encoder
        """
        args = self.args
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location='cpu')
            state_dict = ckpt['model']
            contrast.load_state_dict(ckpt['contrast'])
            encoder1_state_dict = OrderedDict()
            encoder2_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                if 'encoder1' in k:
                    k = k.replace('encoder1.', '')
                    encoder1_state_dict[k] = v
                if 'encoder2' in k:
                    k = k.replace('encoder2.', '')
                    encoder2_state_dict[k] = v
            model.encoder1.load_state_dict(encoder1_state_dict)
            model.encoder2.load_state_dict(encoder2_state_dict)
            print('Pre-trained weights loaded!')
        else:
            print('==============================')
            print('warning: no pre-trained model!')
            print('==============================')

        return model

    def resume_model(self, model, contrast, classifier, optimizer):
        """load classifier checkpoint"""
        args = self.args
        start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model'])
                contrast.load_state_dict(checkpoint['contrast'])
                classifier.load_state_dict(checkpoint['classifier'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        return start_epoch

    def save(self, model, contrast, classifier, optimizer, epoch):
        """save classifier to checkpoint"""
        args = self.args
        if args.local_rank == 0:
            # saving the classifier to each instance
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch))
                torch.save(state, save_file)
                # help release GPU memory
            del state

    def save_seg_models(self, model, classifier, epoch, res):
        args = self.args
        if args.local_rank == 0:
            # saving the classifier to each instance
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'results': res,
            }
            save_file = os.path.join(args.model_folder, 'best_seg.pth')
            torch.save(state, save_file)
            del state

    @staticmethod
    def merge_all_res(x):
        ALIGN_CORNERS=False
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        _x = torch.cat([x[0], x1, x2, x3], 1)
        return _x

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

    def intersection_and_union(self, pred, gt):
        num_classes = self.args.n_class
        mask = gt != 255
        pred = pred[mask].view(-1)
        gt = gt[mask].view(-1)

        intersect = pred[pred == gt]
        area_intersect = torch.histc(intersect.double(), bins=(num_classes), min=0, max=num_classes - 1)
        area_pred_label = torch.histc(pred.double(), bins=(num_classes), min=0, max=num_classes - 1)
        area_label = torch.histc(gt.double(), bins=(num_classes), min=0, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    def eval_seg_iou_acc(self, logits, target):
        logits = logits.detach().cpu()
        target = target.detach().cpu()
        preds = torch.argmax(logits, dim=1)
        batch_size = logits.shape[0]
        total_area_intersect = torch.zeros([self.args.n_class,], dtype=torch.float64)
        total_area_union = torch.zeros([self.args.n_class,], dtype=torch.float64)
        total_area_pred_label = torch.zeros([self.args.n_class,], dtype=torch.float64)
        total_area_label = torch.zeros([self.args.n_class,], dtype=torch.float64)
        for i in range(batch_size):
            area_intersect, area_union, area_pred_label, area_label = self.intersection_and_union(preds[i], target[i])
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
        return total_area_intersect, total_area_union, \
               total_area_pred_label, total_area_label

    def calc_metrics(self, intersect, union, pred_label, label):
        aacc = intersect.sum() / label.sum()
        iou = intersect / union
        acc = intersect / label
        iou[torch.where(torch.isnan(iou))] = 0
        acc[torch.where(torch.isnan(acc))] = 0
        miou = iou.mean()
        macc = acc.mean()
        return aacc, miou, macc, iou, acc

    def eval_seg_aacc(self, logits, target):
        preds = torch.argmax(logits, dim=1)
        same = (preds == target).sum().float()
        all = preds.view(-1).shape[0]
        return same / float(all)

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
        if use_rgb is not None:
            not_use_rgb = torch.logical_not(use_rgb)
            not_use_rgb = not_use_rgb.reshape(bs, 1).repeat(1, num_joints).reshape(bs * num_joints)
            not_use_depth = torch.cat([not_use_rgb, not_use_depth])
        else:
            not_use_depth = torch.cat([torch.zeros_like(not_use_depth), not_use_depth])
        positive_ind[not_use_depth, :] = 0
        positive_ind[:, not_use_depth] = 0

        positive_logits = logsoftmax_logits * positive_ind
        positive_ind_sum = positive_ind.sum(-1)
        positive_ind_sum[positive_ind_sum == 0] = 1
        positive_logits_mean = -positive_logits.sum(-1) / positive_ind_sum
        loss = positive_logits_mean.mean()
        return [loss], [np.array([0])]

    def train_soft_joint_pri3d(self, epoch, train_loader, model, classifier, contrast,   
              criterion_contrast, criterion_pri3d, criterion_seg, optimizer):
        args = self.args
        model.train()
        classifier.train()

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

        loss_jointpairs_rgb = AverageMeter()
        loss_jointpairs_d = AverageMeter()
        loss_jointpairs_joints = AverageMeter()

        loss_scl_meter = AverageMeter()

        seg_loss_meter = AverageMeter()
        seg_aacc_meter = AverageMeter()

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
            label = data[9].cuda(args.gpu, non_blocking=True).long()
            true_label = data[10].cuda(args.gpu, non_blocking=True)

            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # compute feature
            if args.jigsaw:
                raise NotImplementedError
            else:
                _feat1, _feat2, _feat3, f, aux_dict = model(inputs, skeleton, return_fm = True)

            # gather all feature and index
            all_f = self._global_gather(f)
            all_index = self._global_gather(index)

            # loss and metrics
            if args.jigsaw:
                raise NotImplementedError
            else:
                if args.modal.startswith('CMC') or \
                   args.modal.startswith('RGBHHA') or \
                   args.modal.startswith('RGBD'):
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
                        criterion=criterion_contrast, use_depth=use_depth, use_rgb=use_rgb)

                    losses_clip, accuracies_clip = self._compute_soft_pri3d_loss_accuracy(
                        aux_dict['linear_merge1'], aux_dict['linear_merge2'], inputs[:, 3, :, :], criterion_pri3d, use_depth=use_depth, depth_mask=depth_mask, scale=scale)

                    loss_joints, accuracies_joints = self._compute_joints_pri3d_loss_accuracy(
                        aux_dict['linear_merge1'], aux_dict['linear_merge2'], _feat3, criterion_pri3d, original_joints2d, joints_vis, use_depth=use_depth
                    )

                    loss_jointpairs = None

                    loss_scl, _ = self._compute_cross_subject_joints_pri3d_loss(
                        aux_dict['linear_merge1'], aux_dict['linear_merge2'], None, criterion_pri3d, original_joints2d, joints_vis, use_depth=use_depth, index=index, memory=contrast.memory_3, use_rgb=use_rgb
                    )

                    loss = sum(losses) * args.cmc_loss_weights + sum(losses_clip) * args.other_loss_weights + sum(loss_joints) * args.other_loss_weights
                    loss = loss + sum(loss_scl) * args.other_loss_weights

                    if true_label.sum() != 0:
                        if args.supervise_type == 0:
                            linear_merge1 = aux_dict['linear_merge1'][true_label.bool()]
                            linear_merge2 = aux_dict['linear_merge2'][true_label.bool()]
                            linear_merge1 = torch.nn.functional.normalize(linear_merge1, dim=1)
                            linear_merge2 = torch.nn.functional.normalize(linear_merge2, dim=1)
                            max_linear_merge = torch.max(torch.stack([linear_merge1, linear_merge2]), 0)[0]
                            seg_output = classifier(max_linear_merge)
                            loss_seg = criterion_seg[0](seg_output, label[true_label.bool()])
                            aacc = self.eval_seg_aacc(seg_output, label[true_label.bool()])
                        elif args.supervise_type == 1:
                            linear_merge1 = aux_dict['linear_merge1'][true_label.bool()]
                            linear_merge1 = torch.nn.functional.normalize(linear_merge1, dim=1)
                            seg_output = classifier(linear_merge1)
                            loss_seg = criterion_seg[0](seg_output, label[true_label.bool()])
                            aacc = self.eval_seg_aacc(seg_output, label[true_label.bool()])
                        elif args.supervise_type == 2:
                            linear_merge2 = aux_dict['linear_merge2'][true_label.bool()]
                            linear_merge2 = torch.nn.functional.normalize(linear_merge2, dim=1)
                            seg_output = classifier(linear_merge2)
                            loss_seg = criterion_seg[0](seg_output, label[true_label.bool()])
                            aacc = self.eval_seg_aacc(seg_output, label[true_label.bool()])
                        else:
                            tmp = classifier(aux_dict['linear_merge1'])
                            loss_seg = (tmp - tmp).mean()
                            aacc = loss_seg
                        loss += loss_seg * 10
                    else:
                        tmp = classifier(aux_dict['linear_merge1'])
                        loss += (tmp - tmp).mean()

                    update_loss_12 = 0.5 * (losses[0] + losses[1])
                    update_acc_12 = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_23 = 0.5 * (losses[2] + losses[3])
                    update_acc_23 = 0.5 * (accuracies[2] + accuracies[3])
                    update_loss_13 = 0.5 * (losses[4] + losses[5])
                    update_acc_13 = 0.5 * (accuracies[4] + accuracies[5])
                else:
                    raise NotImplementedError

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

            if loss_jointpairs is not None:
                loss_jointpairs_rgb.update(loss_jointpairs[0].item(), bsz)
                loss_jointpairs_d.update(loss_jointpairs[1].item(), bsz)
                loss_jointpairs_joints.update(loss_jointpairs[2].item(), bsz)

            if loss_scl is not None:
                loss_scl_meter.update(loss_scl[0].item(), bsz)

            if true_label.sum() != 0:
                seg_loss_meter.update(loss_seg.item(), bsz)
                seg_aacc_meter.update(aacc.item(), bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}] '
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                          'DT {data_time.val:.3f} ({data_time.avg:.3f}) '
                          'L {loss.val:.3f} ({loss.avg:.3f}) '
                          'a_I {acc1.avg:.3f} {acc2.avg:.3f} {acc3.avg:.3f} '
                          'p3d {loss_rgb2depth.avg:.3f} {acc_rgb2depth.avg:.3f} {loss_depth2rgb.avg:.3f} {acc_depth2rgb.avg:.3f} '
                          'j {loss_rgb2joint.avg:.3f} {acc_rgb2joint.avg:.3f} {loss_d2joint.avg:.3f} {acc_d2joint.avg:.3f} '
                          'jp {loss_jp_rgb.avg:.3f} {loss_jp_d.avg:.3f} {loss_jp_joints.avg:.3f} '
                          'scl {loss_scl_meter.avg:.3f} '
                          'seg {seg_loss_meter.avg:.3f} {seg_aacc_meter.avg:.3f}'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=loss_meter,
                           data_time=data_time, acc1=acc1_meter, acc2=acc2_meter, acc3=acc3_meter,
                           loss_rgb2depth=loss_rgb2depth_meter, loss_depth2rgb=loss_depth2rgb_meter,
                           acc_rgb2depth=acc_rgb2depth_meter, acc_depth2rgb=acc_depth2rgb_meter,
                           loss_rgb2joint=loss_rgb2joint_meter, acc_rgb2joint=acc_rgb2joint_meter,
                           loss_d2joint=loss_d2joint_meter, acc_d2joint=acc_d2joint_meter,
                           loss_jp_rgb=loss_jointpairs_rgb, loss_jp_d=loss_jointpairs_d, loss_jp_joints=loss_jointpairs_joints,
                           loss_scl_meter=loss_scl_meter, seg_loss_meter=seg_loss_meter, seg_aacc_meter=seg_aacc_meter))
                    sys.stdout.flush()

        return seg_loss_meter.avg, seg_aacc_meter.avg

    def validate(self, epoch, val_loader, model, classifier, criterion_seg):
        time1 = time.time()
        args = self.args

        model.eval()
        classifier.eval()

        batch_time = AverageMeter()

        total_area_intersect = torch.zeros([3, self.args.n_class, ], dtype=torch.float64)
        total_area_union = torch.zeros([3, self.args.n_class, ], dtype=torch.float64)
        total_area_pred_label = torch.zeros([3, self.args.n_class, ], dtype=torch.float64)
        total_area_label = torch.zeros([3, self.args.n_class, ], dtype=torch.float64)

        with torch.no_grad():
            end = time.time()
            for idx, data in enumerate(val_loader):
                inputs = data[0].float().cuda(args.gpu, non_blocking=True)
                index = data[1].cuda(args.gpu, non_blocking=True)
                skeleton = data[2].cuda(args.gpu, non_blocking=True)
                original_joints2d = data[4].cuda(args.gpu, non_blocking=True)
                joints_vis = data[5].cuda(args.gpu, non_blocking=True)
                depth_mask = data[7].cuda(args.gpu, non_blocking=True)
                scale = data[8].cuda(args.gpu, non_blocking=True)
                label = data[9].cuda(args.gpu, non_blocking=True)
                true_label = data[10].cuda(args.gpu, non_blocking=True)
                assert torch.all(true_label == 1)

                # compute output
                _feat1, _feat2, _feat3, f, aux_dict = model(inputs, skeleton, return_fm = True)
                
                linear_merge1 = aux_dict['linear_merge1']
                linear_merge2 = aux_dict['linear_merge2']
                linear_merge1 = torch.nn.functional.normalize(linear_merge1, dim=1)
                linear_merge2 = torch.nn.functional.normalize(linear_merge2, dim=1)
                max_linear_merge = torch.max(torch.stack([linear_merge1, linear_merge2]), 0)[0]
                
                rgbd_output = classifier(max_linear_merge)
                rgbd_loss_seg = criterion_seg[0](rgbd_output, label)
                
                rgb_output = classifier(linear_merge1)
                rgb_loss_seg = criterion_seg[0](rgb_output, label)

                d_output = classifier(linear_merge2)
                d_loss_seg = criterion_seg[0](d_output, label)

                for i, output in enumerate([rgb_output, d_output, rgbd_output]):
                    area_intersect, area_union, area_pred_label, area_label = self.eval_seg_iou_acc(output, label)
                    total_area_intersect[i] += area_intersect
                    total_area_union[i] += area_union
                    total_area_pred_label[i] += area_pred_label
                    total_area_label[i] += area_label
                    
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.local_rank == 0 and idx % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                           idx, len(val_loader), batch_time=batch_time))

            total_area_intersect[0], total_area_union[0], \
            total_area_pred_label[0], total_area_label[0] = self.gather_eval_counter(
                total_area_intersect[0], total_area_union[0],
                total_area_pred_label[0], total_area_label[0]
            )
            dist.barrier()
            total_area_intersect[1], total_area_union[1], \
            total_area_pred_label[1], total_area_label[1] = self.gather_eval_counter(
                total_area_intersect[1], total_area_union[1],
                total_area_pred_label[1], total_area_label[1]
            )
            dist.barrier()
            total_area_intersect[2], total_area_union[2], \
            total_area_pred_label[2], total_area_label[2] = self.gather_eval_counter(
                total_area_intersect[2], total_area_union[2],
                total_area_pred_label[2], total_area_label[2]
            )

            aacc1, miou1, macc1, iou1, acc1 = self.calc_metrics(total_area_intersect[0],
                                                                total_area_union[0],
                                                                total_area_pred_label[0],
                                                                total_area_label[0])
            aacc2, miou2, macc2, iou2, acc2 = self.calc_metrics(total_area_intersect[1],
                                                                total_area_union[1],
                                                                total_area_pred_label[1],
                                                                total_area_label[1])
            aacc3, miou3, macc3, iou3, acc3 = self.calc_metrics(total_area_intersect[2],
                                                                total_area_union[2],
                                                                total_area_pred_label[2],
                                                                total_area_label[2])

            if self.args.rank == 0:
                self.mprint('name            \t\t iou1   \t\t acc1   \t\t iou2   \t\t acc2   \t\t iou3   \t\t acc3  ')
                for i, class_name in enumerate(classes):
                    self.mprint(f'{class_name:15s} \t\t {iou1[i]:.4f} \t\t {acc1[i]:.4f} \t\t {iou2[i]:.4f} \t\t {acc2[i]:.4f} \t\t {iou3[i]:.4f} \t\t {acc3[i]:.4f}')

                self.mprint(f'mean           \t\t {miou1:.4f} \t {macc1:.4f} \t {miou2:.4f} \t {macc2:.4f} \t {miou3:.4f} \t {macc3:.4f}')
                self.mprint(f'All Acc: {aacc1:.4f} \t {aacc2:.4f} \t {aacc3:.4f}')

                time2 = time.time()
                self.mprint('eval epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return {
            'rgb':  {'aacc': aacc1, 'miou': miou1, 'macc': macc1, 'iou': iou1, 'acc': acc1},
            'd':    {'aacc': aacc2, 'miou': miou2, 'macc': macc2, 'iou': iou2, 'acc': acc2},
            'rgbd': {'aacc': aacc3, 'miou': miou3, 'macc': macc3, 'iou': iou3, 'acc': acc3}
        }

