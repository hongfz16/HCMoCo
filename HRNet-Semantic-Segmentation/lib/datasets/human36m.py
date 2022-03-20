# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
 
# 0  : background
# 1  : right hip
# 2  : right knee
# 3  : right foot
# 6  : left hip
# 7  : left knee
# 8  : left foot
# 17 : left shoulder
# 18 : left elbow
# 19 : left hand
# 25 : right shoulder
# 26 : right elbow
# 27 : right hand
# 32 : crotch
# 33 : right thigh
# 34 : right calf
# 38 : left thigh
# 39 : left calf
# 43 : lower spine
# 44 : upper spine
# 46 : head
# 49 : left arm
# 50 : left forearm
# 56 : right arm
# 58 : right forearm

# 1, 6
# 2, 7
# 3, 8
# 17, 25
# 18, 26
# 19, 27
# 33, 38
# 34, 39
# 49, 56
# 50, 58

class Human36M(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=25,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=473, 
                 crop_size=(473, 473), 
                 downsample_rate=1,
                 scale_factor=11,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Human36M, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.left_right_pairs = np.array(
            [[1, 6],
            [2, 7],
            [3, 8],
            [17, 25],
            [18, 26],
            [19, 27],
            [33, 38],
            [34, 39],
            [49, 56],
            [50, 58]]
        )

        self.original_label = np.array([0, 1, 2, 3, 6, 7, 8, 17, 18, 19, 25, 26, 27, 32, 33, 34, 38, 39, 43, 44, 46, 49, 50, 56, 58])
        self.label_mapper = np.arange(60)
        for i, l in enumerate(self.original_label):
            self.label_mapper[l] = i
        self.mapped_left_right_pairs = self.label_mapper[self.left_right_pairs]

        self.multi_scale = multi_scale
        self.flip = flip
        with open(self.list_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            self.img_list = [os.path.join(self.root, l) for l in lines]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
    
    def read_files(self):
        files = []
        for image_path in self.img_list:
            splitpath = image_path.split('/')
            splitpath[-4] = 'seg'
            label_path = '/'.join(splitpath)
            name = '/'.join(splitpath[-3:])
            sample = {
                'img': image_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
        return files

    def resize_image(self, image, label, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
     
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
         
        image = cv2.imread(item['img'], cv2.IMREAD_COLOR)
        label = np.array(Image.open(item['label']))
        label = self.label_mapper[label]

        size = label.shape

        if 'eval' in self.list_path:
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, (1000, 1000), interpolation = cv2.INTER_LINEAR_EXACT)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :] 
            label = label[:, ::flip]

            if flip == -1:
                left_idx = self.mapped_left_right_pairs[:, 0].reshape(-1)
                right_idx = self.mapped_left_right_pairs[:, 1].reshape(-1)
                for i in range(0, self.mapped_left_right_pairs.shape[0]):
                    right_pos = np.where(label == right_idx[i])
                    left_pos = np.where(label == left_idx[i])
                    label[right_pos[0], right_pos[1]] = left_idx[i]
                    label[left_pos[0], left_pos[1]] = right_idx[i]
        
        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, 
                                self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred, 
                          size=(size[-2], size[-1]), 
                          mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_output = flip_output.cpu().numpy()
            flip_pred = flip_output.copy()
            for pair in self.mapped_left_right_pairs:
                flip_pred[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
                flip_pred[:, pair[1], :, :] = flip_output[:, pair[0], :, :]
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for k, v in enumerate(self.label_mapper):
                label[temp == k] = v
        else:
            for v, k in enumerate(self.label_mapper):
                label[temp == k] = v
        return label

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_gt(self, images, preds, sv_path1, sv_path2, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy().astype(np.uint8)
        # preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = ((images * self.std) + self.mean) * 255
        images = images.astype(np.uint8)
        for i in range(preds.shape[0]):
            # import pdb; pdb.set_trace()
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path2, '_'.join(name[i].split('/'))))

            ori_img = Image.fromarray(images[i])
            ori_img.save(os.path.join(sv_path1, '_'.join(name[i].split('/'))))

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, '_'.join(name[i].split('/'))))
