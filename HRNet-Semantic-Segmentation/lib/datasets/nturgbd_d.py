# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import random

import cv2
import numpy as np

from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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

class NTURGBDD(BaseDataset):
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
                 mean=[0, 0, 0], 
                 std=[1, 1, 1]):

        super(NTURGBDD, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path

        # self.class_weights = None
        # v1 train statistics
        # weights = [1.453,49.297,49.219,47.848,49.313,49.235,47.818,49.520,49.960,49.140,49.468,49.995,49.167,49.194,45.048,46.354,45.038,46.415,43.543,41.302,43.245,48.408,48.542,48.298,48.659]
        weights = [1.448,49.234,49.483,48.030,49.247,49.492,48.018,49.704,50.052,49.369,49.694,50.090,49.425,49.459,45.846,47.156,45.868,47.197,44.167,42.789,44.341,48.632,48.873,48.644,49.004]
        self.class_weights = torch.from_numpy(np.array(weights).astype(np.float32))

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
            splitpath[-2] = 'png_annotation_v2'
            label_path = '/'.join(splitpath)
            label_path = label_path[:-3] + 'png'
            name = splitpath[-1]
            splitpath[-2] = 'depth'
            splitpath[-1] = 'MDepth-' + splitpath[-1][:-3] + 'png'
            depth_path = '/'.join(splitpath)
            sample = {
                'img': depth_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
        return files

    def resize_image(self, image, label, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_NEAREST) 
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def process_depth_map(self, image):
        image = image.astype(np.float32) / 1000.0
        image = np.stack([image, image, image], -1)

        if (image != 0).sum() == 0:
            mean = 0
        else:
            mean = image.sum() / (image != 0).sum()
        image[image != 0] = image[image != 0] - mean

        image = image.transpose((2, 0, 1))

        return image

    def rand_crop(self, image, label):
        h, w = image.shape
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def gen_sample(self, image, label, multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, 
                                                    rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image, 
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        image = self.process_depth_map(image)
        label = self.label_transform(label)
        
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, 
                               None, 
                               fx=self.downsample_rate,
                               fy=self.downsample_rate, 
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        image = cv2.imread(item['img'], -1).astype(np.uint16)
        label = np.array(Image.open(item['label']))
        label = self.label_mapper[label]

        size = label.shape

        if 'val' in self.list_path:
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_NEAREST)
            # label = cv2.resize(label, (1000, 1000), interpolation = cv2.INTER_LINEAR_EXACT)
            label = cv2.resize(label, (1000, 1000), interpolation = cv2.INTER_NEAREST)
            image = self.process_depth_map(image)
            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip] 
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

    # def save_pred(self, images, preds, sv_path1, sv_path2, name):
    #     palette = self.get_palette(256)
    #     preds = preds.cpu().numpy().copy().astype(np.uint8)
    #     # preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
    #     images = images.permute(0, 2, 3, 1).cpu().numpy()
    #     # images = ((images * self.std) + self.mean) * 255
    #     # images = images.astype(np.uint8)
    #     for i in range(preds.shape[0]):
    #         # import pdb; pdb.set_trace()
    #         pred = self.convert_label(preds[i], inverse=True)
    #         save_img = Image.fromarray(pred)
    #         save_img.putpalette(palette)
    #         save_img.save(os.path.join(sv_path2, ('_'.join(name[i].split('/'))[:-3] + 'png')))

    #         plt.axis('off')
    #         plt.imshow(images[i][:, :, 0])
    #         plt.savefig(os.path.join(sv_path1, ('_'.join(name[i].split('/'))[:-3] + 'png')), bbox_inches='tight', pad_inches=0)

    #         # ori_img = Image.fromarray(images[i][:, :, 0])
    #         # ori_img.save(os.path.join(sv_path1, ('_'.join(name[i].split('/'))[:-3] + 'png')))

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            # pred = preds[i]
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, ('_'.join(name[i].split('/'))[:-3] + 'png')))
