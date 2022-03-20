from __future__ import print_function

import numpy as np
import torch
from torchvision import datasets
import PIL
import cv2

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index

import cv2
import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
import pickle

import random
import json_tricks as json
import copy
from .mpii_utils import get_affine_transform, affine_transform

import re

class NTURGBD(data.Dataset):
    def __init__(self, root, file_list, size=320, random_flip=False, random_resized_crop=False, use_jigsaw=False, need_gt=False):
        super(NTURGBD, self).__init__()
        self.root = root
        self.file_list = [f.strip() for f in open(file_list, 'r').readlines()]

        self.scale = (0.8, 1.2)
        # self.scale = (1, 1)
        self.ratio = (3./4, 4./3)
        # self.ratio = (1, 1)
        self.size = (size, size)
        
        self.random_flip = random_flip
        self.random_resized_crop = random_resized_crop
        self.use_jigsaw = use_jigsaw
        self.need_gt = need_gt

        assert not self.use_jigsaw
        assert not self.need_gt

        def transfer_fname(f, replace_prefix='HumanRGBD/NTURGBD/nturgb+d_depth_masked'):
            # f = f.replace('nturgb+d_rgb_warped', 'nturgb+d_depth_masked')
            f = f.replace('nturgb+d_rgb_warped_correction', replace_prefix)
            f = f.replace('WRGB', 'MDepth')
            f = f.replace('jpg', 'png')
            return f

        self.image_list = [os.path.join(self.root, f) for f in self.file_list]
        self.depth_list = [os.path.join(self.root, transfer_fname(f)) for f in self.file_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index, return_resize_param=False):
        image_path = self.image_list[index]
        depth_path = self.depth_list[index]
        img = Image.open(image_path).convert('RGB')
        depth = cv2.imread(depth_path, -1).astype(np.uint16)
        ind = np.where(depth > 0)
        xmin, xmax = ind[0].min(), ind[0].max()
        ymin, ymax = ind[1].min(), ind[1].max()

        depth = Image.fromarray(depth)

        if self.random_resized_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=self.ratio
            )
            mid_x = i + h / 2.0
            mid_y = j + w / 2.0

            if mid_x < xmin:
                new_mid_x = xmin
            elif mid_x > xmax:
                new_mid_x = xmax
            else:
                new_mid_x = mid_x

            if mid_y < ymin:
                new_mid_y = ymin
            elif mid_y > ymax:
                new_mid_y = ymax
            else:
                new_mid_y = mid_y

            i = int(new_mid_x - h / 2.0)
            j = int(new_mid_y - w / 2.0)

            img = TF.resized_crop(img, i, j, h, w, self.size)
            depth = TF.resized_crop(depth, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
        else:
            i, j, h, w = 0, 0, img.size[0], img.size[1]

        need_flip = random.random() >= 0.5
        if self.random_flip and need_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img /= 255.0
        img -= torch.from_numpy(np.array([0.485, 0.456, 0.406]))
        img /= torch.from_numpy(np.array([0.229, 0.224, 0.225]))

        depth = torch.from_numpy(np.array(depth).astype(np.float32) / 1000.0)
        depth = torch.stack([depth, depth, depth], dim=0)

        if return_resize_param:
            return torch.cat([
                img.permute(2, 0, 1),
                depth,
            ], dim=0), index, (i, j, h, w, need_flip)

        return torch.cat([
            img.permute(2, 0, 1),
            depth,
        ], dim=0), index

class NTURGBD3DSkeleton(NTURGBD):
    def __init__(self, root, file_list, size=320, random_flip=False, random_resized_crop=False, use_jigsaw=False, need_gt=False):
        super(NTURGBD3DSkeleton, self).__init__(root, file_list, size, random_flip, random_resized_crop, use_jigsaw, need_gt)
        def transfer_fname(f, replace_prefix='HumanRGBD/NTURGBD/nturgb+d_parsed_skeleton'):
            # f = f.replace('nturgb+d_rgb_warped', 'nturgb+d_parsed_skeleton')
            f = f.replace('nturgb+d_rgb_warped_correction', replace_prefix)
            f = f.replace('WRGB', 'Skeleton')
            f = f.replace('jpg', 'pkl')
            num = int(f[-12:-4])
            f = f[:-12] + str(num - 1).zfill(8) + f[-4:]
            return f
        self.skeleton_list = [os.path.join(self.root, transfer_fname(f)) for f in self.file_list]

    def __getitem__(self, index, return_resize_param=False):
        # if return_resize_param:
        #     rgbd, index, resize_param = super().__getitem__(index, True)
        # else:
        #     rgbd, index = super().__getitem__(index, False)

        image_path = self.image_list[index]
        depth_path = self.depth_list[index]
        img = Image.open(image_path).convert('RGB')
        depth = cv2.imread(depth_path, -1).astype(np.uint16)
        depth = Image.fromarray(depth)

        original_h, original_w = img.size[1], img.size[0]

        skeleton_fname = self.skeleton_list[index]
        with open(skeleton_fname, 'rb') as f:
            skeleton_dict = pickle.load(f)
        # assert len(skeleton_dict['joints']) == 1, print(len(skeleton_dict['joints']), skeleton_fname)
        joints3d = []
        for j in skeleton_dict['joints'][0]['3d_loc']:
            joints3d.append(j)
        root_joint = np.array(joints3d[0])

        # print(">>>>>>>>>joints3d")
        # print(joints3d)
        # print(">>>>>>>>>joints_depth")
        # jointsdepth = []
        # joints2d = []
        # for joint in skeleton_dict['joints'][0]['d_loc']:
        #     joints2d.append(joint)
        #     jointsdepth.append(np.array(depth)[int(joint[1]), int(joint[0])] / 1000.0)
        # print(jointsdepth)
        # print(">>>>>>>>>joints2d")
        # print(joints2d)
        
        joints3d = torch.from_numpy(np.array(joints3d, dtype=np.float32) - root_joint)

        if self.random_resized_crop:
            joints2d = []
            for j in skeleton_dict['joints'][0]['d_loc']:
                joints2d.append(j)
            joints2d = np.array(joints2d)
            assert not np.any(np.isnan(joints2d)), skeleton_fname
            human_min_x, human_max_x = joints2d[:, 1].min(), joints2d[:, 1].max()
            human_min_y, human_max_y = joints2d[:, 0].min(), joints2d[:, 0].max()
            rand_x = random.randrange(int(human_min_x), int(human_max_x))
            rand_y = random.randrange(int(human_min_y), int(human_max_y))
            _, _, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.08, 1.2), ratio=(1, 1)
            )
            i = int(rand_x - h / 2.0)
            j = int(rand_y - w / 2.0)
            img = TF.resized_crop(img, i, j, h, w, self.size)
            depth = TF.resized_crop(depth, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
        else:
            i, j, h, w = 0, 0, img.size[0], img.size[1]

        need_flip = random.random() >= 0.5
        if self.random_flip and need_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img /= 255.0
        img -= torch.from_numpy(np.array([0.485, 0.456, 0.406]))
        img /= torch.from_numpy(np.array([0.229, 0.224, 0.225]))

        depth = torch.from_numpy(np.array(depth).astype(np.float32) / 1000.0)
        depth = torch.stack([depth, depth, depth], dim=0)

        rgbd = torch.cat([img.permute(2, 0, 1), depth], dim=0)
        resize_param = (i, j, h, w, need_flip, original_h, original_w)

        if return_resize_param:
            return rgbd, index, joints3d, resize_param, skeleton_dict
        return rgbd, index, joints3d

class NTURGBD3D2DSkeleton(NTURGBD3DSkeleton):
    def __init__(self, root, file_list, size=320, random_flip=False, random_resized_crop=False, use_jigsaw=False, need_gt=False):
        super(NTURGBD3D2DSkeleton, self).__init__(root, file_list, size, random_flip, random_resized_crop, use_jigsaw, need_gt)
        self.sigma = 2
        self.num_joints = 25
        pos_enc = np.zeros([self.num_joints, 3])
        for i in range(1, self.num_joints + 1):
            pos_enc[i - 1, 0] = (i % 3)
            pos_enc[i - 1, 1] = (i // 3) % 3
            pos_enc[i - 1, 2] = (i // 9) % 3
        self.pos_enc_kinect = pos_enc * 0.5
    
    def generate_joint2d_heatmap(self, joints2d, num_joints, image_h, image_w, pos_enc):
        heatmap = np.zeros([num_joints, image_h, image_w])
        for i in range(num_joints):
            mu_x = joints2d[i, 0]
            mu_y = joints2d[i, 1]
            x = np.arange(0, image_w, 1, np.float32)
            y = np.arange(0, image_h, 1, np.float32)
            y = y[:, None]
            heatmap[i] = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))
        rgb_heatmap = pos_enc.reshape(-1, 3, 1, 1) * np.tile(heatmap, (3, 1, 1, 1)).transpose(1, 0, 2, 3)
        rgb_heatmap = np.amax(rgb_heatmap, axis=0).transpose(1, 2, 0)
        return rgb_heatmap
    
    def transform_heatmap(self, rgb_heatmap, resize_param):
        scale = 10000
        rgb_heatmap = (rgb_heatmap * scale).astype(np.uint16)
        rgb_heatmap_list = []
        i, j, h, w, need_flip, _, _ = resize_param
        for dim in range(3):
            rgb_heatmap_i = Image.fromarray(rgb_heatmap[:, :, dim])
            if self.random_resized_crop:
                rgb_heatmap_i = TF.resized_crop(rgb_heatmap_i, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
            if need_flip and self.random_flip:
                rgb_heatmap_i = rgb_heatmap_i.transpose(Image.FLIP_LEFT_RIGHT)
            rgb_heatmap_list.append(np.array(rgb_heatmap_i))
        rgb_heatmap = np.stack(rgb_heatmap_list, axis=-1)
        rgb_heatmap = torch.from_numpy(rgb_heatmap.astype(np.float32) / float(scale))
        return rgb_heatmap

    def __getitem__(self, index):
        rgbd, index, joints3d, resize_param, skeleton_dict = super().__getitem__(index, return_resize_param=True)
        joints2d = []
        for joint in skeleton_dict['joints'][0]['d_loc']:
            joints2d.append(joint)
        joints2d = np.array(joints2d, dtype=np.float32)
        num_joints = joints2d.shape[0]
        assert num_joints == self.num_joints
        image_h, image_w = rgbd.shape[-2], rgbd.shape[-1]
        rgb_heatmap = self.generate_joint2d_heatmap(joints2d, num_joints, image_h, image_w, self.pos_enc_kinect)
        rgb_heatmap = self.transform_heatmap(rgb_heatmap, resize_param)
        return torch.cat([rgbd, rgb_heatmap.permute(2, 0, 1)], dim=0), index, joints3d

class NTUMPIIRGBD3D2DSkeleton(NTURGBD3D2DSkeleton):
    def __init__(self, ntu_root, ntu_file_list, mpii_root, mpii_image_set, size=320, random_flip=False, random_resized_crop=False,
                 use_jigsaw=False, need_gt=False):
        super(NTUMPIIRGBD3D2DSkeleton, self).__init__(ntu_root, ntu_file_list, size, random_flip, random_resized_crop, use_jigsaw, need_gt)
        self.mpii_root = mpii_root
        self.mpii_image_set = mpii_image_set
        
        self.mpii_num_joints = 16
        pos_enc = np.zeros([self.mpii_num_joints, 3])
        for i in range(1, self.mpii_num_joints + 1):
            pos_enc[i - 1, 0] = (i % 3)
            pos_enc[i - 1, 1] = (i // 3) % 3
            pos_enc[i - 1, 2] = (i // 9) % 3
        self.pos_enc_mpii = pos_enc * 0.5

        self.mpii_num_joints_half_body = 8
        self.mpii_data_format = 'jpg'
        self.db = self._get_db(mpii_root, mpii_image_set, self.mpii_num_joints, self.mpii_data_format)

    def Kinect2MPII(self, joints):
        # select_ind = [18, 17, 16, 12, 13, 14, 0, 1, 2, 3, 10, 9, 8, 4, 5, 6]
        select_ind = [14, 13, 12, 16, 17, 18, 0, 1, 2, 3, 6, 5, 4, 8, 9, 10]
        return joints[select_ind].reshape(16, 2)

    def _get_db(self, root, image_set, num_joints, data_format):
        # create train/val split
        file_name = os.path.join(
            root, 'annot', image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((num_joints,  3), dtype=np.float)
            if image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def __len__(self):
        return len(self.db) + len(self.image_list)

    def mpii_getitem(self, index):
        db_rec = copy.deepcopy(self.db[index])
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.random_resized_crop:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() < 0.6 else 0
        trans = get_affine_transform(c, s, r, self.size)
        img = cv2.warpAffine(
            data_numpy,
            trans,
            self.size,
            flags = cv2.INTER_LINEAR)
        for i in range(self.mpii_num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        rgb_heatmap = self.generate_joint2d_heatmap(joints[:, :2], self.mpii_num_joints, self.size[0], self.size[1], self.pos_enc_mpii)

        if self.random_flip and random.random() <= 0.5:
            img = np.array(img, dtype=np.float32)[:, ::-1, :]
            rgb_heatmap = rgb_heatmap[:, ::-1, :]

        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img /= 255.0
        img -= torch.from_numpy(np.array([0.485, 0.456, 0.406]))
        img /= torch.from_numpy(np.array([0.229, 0.224, 0.225]))
        rgb_heatmap = torch.from_numpy(rgb_heatmap.astype(np.float32))

        
        img = img.permute(2, 0, 1)
        rgb_heatmap = rgb_heatmap.permute(2, 0, 1)

        fake_depth = torch.zeros_like(img)

        return torch.cat([img, fake_depth, rgb_heatmap], 0)

    def __getitem__(self, index):
        if index < len(self.db):
            data = self.mpii_getitem(index)
            joints3d = torch.zeros([self.num_joints, 3])
            true_depth = 0
        else:
            rgbd, _, joints3d, resize_param, skeleton_dict = NTURGBD3DSkeleton.__getitem__(self, index - len(self.db), return_resize_param=True)
            joints2d = []
            for joint in skeleton_dict['joints'][0]['d_loc']:
                joints2d.append(joint)
            joints2d = np.array(joints2d, dtype=np.float32)
            joints2d = self.Kinect2MPII(joints2d)
            num_joints = joints2d.shape[0]
            assert num_joints == self.mpii_num_joints
            image_h, image_w = rgbd.shape[-2], rgbd.shape[-1]
            rgb_heatmap = self.generate_joint2d_heatmap(joints2d, num_joints, image_h, image_w, self.pos_enc_mpii)
            rgb_heatmap = self.transform_heatmap(rgb_heatmap, resize_param)
            data = torch.cat([rgbd, rgb_heatmap.permute(2, 0, 1)], dim=0)
            true_depth = 1
        
        return data, index, joints3d, true_depth

def generate_scale_mpii(joint2d, joint_vis):
    # MPII: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
    #       5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
    #       10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder,
    #       14 - l elbow, 15 - l wrist
    # reference_pair = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [10, 11], [11, 12], [13, 14], [14, 15]]
    # reference_pair = [p for p in reference_pair if joint_vis[p[0]] and joint_vis[p[1]]]
    num_joints = joint2d.shape[0]
    pairwise_dist = joint2d.reshape(num_joints, 1, 2) - joint2d.reshape(1, num_joints, 2)
    pairwise_dist = np.sqrt((pairwise_dist ** 2).sum(-1))
    pairwise_dist[~joint_vis, :] = -1
    pairwise_dist[:, ~joint_vis] = -1
    max_dist = pairwise_dist.max()
    if max_dist == -1 or max_dist == 0:
        return 80
    return max_dist

class NTUMPIIRGBD3D2DSkeletonGCN(NTUMPIIRGBD3D2DSkeleton):
    def __init__(self, ntu_root, ntu_file_list, mpii_root, mpii_image_set, size=320, random_flip=False, random_resized_crop=False,
                 use_jigsaw=False, need_gt=False):
        super(NTUMPIIRGBD3D2DSkeletonGCN, self).__init__(ntu_root, ntu_file_list, mpii_root, mpii_image_set, size=size,
                                                         random_flip=random_flip, random_resized_crop=random_resized_crop,
                                                         use_jigsaw=use_jigsaw, need_gt=need_gt)
        self.flip_pairs = [[0,5],[1,4],[2,3],[10,15],[11,14],[12,13]]

    def normalize_joints_myway(self, _joints2d, root_index=6):
        joints2d = _joints2d.copy()
        joints2d -= joints2d[root_index, :]
        joints2d = joints2d[:, ::-1]
        s = max(joints2d.max(), np.abs(joints2d.min()))
        joints2d /= s
        return joints2d

    def normalize_joints_liftway(self, _joints2d, w, h):
        _joints2d = _joints2d[:, ::-1]
        return _joints2d / w * 2 - np.array([1, h / w])

    def flip_normalized_joints(self, norm_joints):
        norm_joints[:, 1] = -norm_joints[:, 1]
        tmp_joints = norm_joints.copy()
        for i, j in self.flip_pairs:
            norm_joints[i, :] = tmp_joints[j, :]
            norm_joints[j, :] = tmp_joints[i, :]
        return norm_joints

    def mpii_getitem(self, index):
        db_rec = copy.deepcopy(self.db[index])
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.random_resized_crop:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() < 0.6 else 0
        trans = get_affine_transform(c, s, r, self.size)
        img = cv2.warpAffine(
            data_numpy,
            trans,
            self.size,
            flags = cv2.INTER_LINEAR)

        original_joints = joints[:, :2].copy()
        if self.random_resized_crop:
            num_joints = joints_vis.shape[0]
            for i in range(num_joints):
                if joints_vis[i, 0] > 0.0:
                    original_joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        norm_joints = self.normalize_joints_myway(joints[:, :2])

        original_joints = original_joints[:, ::-1]
        # rgb_heatmap = self.generate_joint2d_heatmap(joints[:, :2], self.mpii_num_joints, self.size[0], self.size[1], self.pos_enc_mpii)

        if self.random_flip and random.random() <= 0.5:
            img = np.array(img, dtype=np.float32)[:, ::-1, :]
            norm_joints = self.flip_normalized_joints(norm_joints)
            original_joints[:, 1] = self.size[1] - original_joints[:, 1]
            # rgb_heatmap = rgb_heatmap[:, ::-1, :]

        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img /= 255.0
        img -= torch.from_numpy(np.array([0.485, 0.456, 0.406]))
        img /= torch.from_numpy(np.array([0.229, 0.224, 0.225]))
        # rgb_heatmap = torch.from_numpy(rgb_heatmap.astype(np.float32))

        img = img.permute(2, 0, 1)
        # rgb_heatmap = rgb_heatmap.permute(2, 0, 1)
        fake_depth = torch.zeros_like(img)

        joints_vis = np.logical_and(
                        np.logical_and(np.logical_and(original_joints[:, 0] >= 0, original_joints[:, 0] < self.size[0]),
                                       np.logical_and(original_joints[:, 1] >= 0, original_joints[:, 1] < self.size[0])),
                        joints_vis[:, 0])

        return torch.cat([img, fake_depth], 0), norm_joints, original_joints, joints_vis

    def transfer_fname_geodesic(self, f):
        f = f.replace('nturgb+d_rgb_warped_correction', 'HumanRGBD/NTURGBD/nturgb+d_geodesic_masked')
        f = f.replace('WRGB', 'Geo')
        f = f.replace('jpg', 'pkl')
        return f

    def __getitem__(self, index):
        if index < len(self.db):
            rgbd, norm_joints, original_joints2d, joints_vis = self.mpii_getitem(index)
            joints3d = torch.zeros([self.num_joints, 3])
            true_depth = 0
            depth_mask = torch.zeros_like(rgbd[0, :, :])
        else:
            rgbd, _, joints3d, resize_param, skeleton_dict = NTURGBD3DSkeleton.__getitem__(self, index - len(self.db), return_resize_param=True)
            joints2d = []
            for joint in skeleton_dict['joints'][0]['d_loc']:
                joints2d.append(joint)
            joints2d = np.array(joints2d, dtype=np.float32)
            joints2d = self.Kinect2MPII(joints2d)
            num_joints = joints2d.shape[0]
            assert num_joints == self.mpii_num_joints

            i, j, h, w, need_flip, original_h, original_w = resize_param

            norm_joints = self.normalize_joints_myway(joints2d)
            if self.random_flip and resize_param[-1]:
                norm_joints = self.flip_normalized_joints(norm_joints)

            joints_vis = np.logical_and(np.logical_and(joints2d[:, 1] > i, joints2d[:, 1] < i + h),
                                        np.logical_and(joints2d[:, 0] > j, joints2d[:, 1] < j + w))
            original_joints2d = joints2d[:, ::-1].copy()
            original_joints2d[:, 0] = (original_joints2d[:, 0] - i) / h * self.size[0]
            original_joints2d[:, 1] = (original_joints2d[:, 1] - j) / w * self.size[0]
            true_depth = 1

            depth = rgbd[3, :, :]
            depth_mask = depth > 0
            
            mean = depth.sum() / depth_mask.sum()
            n = depth_mask.sum()
            std = torch.sqrt((((depth - mean) ** 2) * depth_mask).sum() / (n - 1))
            norm_depth = (depth - mean)
            norm_depth[~depth_mask] = 0
            rgbd[3:, :, :] = norm_depth.unsqueeze(0)

        original_joints2d[np.logical_not(joints_vis), :] = 0
        norm_joints[np.logical_not(joints_vis), :] = 0

        scale = generate_scale_mpii(original_joints2d, joints_vis)

        return rgbd, index, torch.from_numpy(norm_joints.copy().astype(np.float32)), \
               joints3d, torch.from_numpy(original_joints2d.copy()), \
               torch.from_numpy(joints_vis.astype(np.int32).copy()), true_depth, \
               depth_mask.float(), scale
            #    i, j, h, w, original_h, original_w

from pycocotools.coco import COCO

class NTUCOCORGBD3D2DSkeletonGCN(NTURGBD3D2DSkeleton):
    def __init__(self, ntu_root, ntu_file_list, coco_root, coco_image_set, size=320, random_flip=False, random_resized_crop=False,
                 use_jigsaw=False, need_gt=False):
        super(NTUCOCORGBD3D2DSkeletonGCN, self).__init__(ntu_root, ntu_file_list, size=size,
                                                         random_flip=random_flip, random_resized_crop=random_resized_crop,
                                                         use_jigsaw=use_jigsaw, need_gt=need_gt)
        self.coco_root = coco_root
        self.coco_image_set = coco_image_set

        self.coco = COCO(self._get_ann_file_keypoint())
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        print('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )
        self.image_set_index = self._load_image_set_index()

        self.num_images = len(self.image_set_index)
        print('=> num_images: {}'.format(self.num_images))

        self.coco_num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.coco_num_joints, 1))

        self.is_train = True
        self.aspect_ratio = 1.0
        self.pixel_std = 200
        self.data_format = 'jpg'

        self.db = self._get_db()

    def __len__(self):
        return len(self.db) + len(self.image_list)

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.coco_image_set else 'image_info'
        return os.path.join(
            self.coco_root,
            'annotations',
            prefix + '_' + self.coco_image_set + '.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            raise NotImplementedError
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.coco_num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.coco_num_joints, 3), dtype=np.float)
            for ipt in range(self.coco_num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.coco_image_set:
            file_name = 'COCO_%s_' % self.coco_image_set + file_name

        prefix = 'test2017' if 'test' in self.coco_image_set else self.coco_image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.coco_root, 'images', data_name, file_name)

        return image_path

    def normalize_joints_myway(self, _joints2d, root_index=6):
        joints2d = _joints2d.copy()
        joints2d -= joints2d[root_index, :]
        joints2d = joints2d[:, ::-1]
        s = max(joints2d.max(), np.abs(joints2d.min()))
        joints2d /= s
        return joints2d

    def normalize_joints_liftway(self, _joints2d, w, h):
        _joints2d = _joints2d[:, ::-1]
        return _joints2d / w * 2 - np.array([1, h / w])

    def flip_normalized_joints(self, norm_joints):
        norm_joints[:, 1] = -norm_joints[:, 1]
        tmp_joints = norm_joints.copy()
        for i, j in self.flip_pairs:
            norm_joints[i, :] = tmp_joints[j, :]
            norm_joints[j, :] = tmp_joints[i, :]
        return norm_joints

    def coco_getitem(self, index):
        db_rec = copy.deepcopy(self.db[index])
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.random_resized_crop:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() < 0.6 else 0
        trans = get_affine_transform(c, s, r, self.size)
        img = cv2.warpAffine(
            data_numpy,
            trans,
            self.size,
            flags = cv2.INTER_LINEAR)

        original_joints = joints[:, :2].copy()
        if self.random_resized_crop:
            num_joints = joints_vis.shape[0]
            for i in range(num_joints):
                if joints_vis[i, 0] > 0.0:
                    original_joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        norm_joints = self.normalize_joints_myway(joints[:, :2])

        original_joints = original_joints[:, ::-1]
        # rgb_heatmap = self.generate_joint2d_heatmap(joints[:, :2], self.mpii_num_joints, self.size[0], self.size[1], self.pos_enc_mpii)

        if self.random_flip and random.random() <= 0.5:
            img = np.array(img, dtype=np.float32)[:, ::-1, :]
            norm_joints = self.flip_normalized_joints(norm_joints)
            original_joints[:, 1] = self.size[1] - original_joints[:, 1]
            # rgb_heatmap = rgb_heatmap[:, ::-1, :]

        img = torch.from_numpy(np.array(img, dtype=np.float32))
        img /= 255.0
        img -= torch.from_numpy(np.array([0.485, 0.456, 0.406]))
        img /= torch.from_numpy(np.array([0.229, 0.224, 0.225]))
        # rgb_heatmap = torch.from_numpy(rgb_heatmap.astype(np.float32))

        img = img.permute(2, 0, 1)
        # rgb_heatmap = rgb_heatmap.permute(2, 0, 1)
        fake_depth = torch.zeros_like(img)

        joints_vis = np.logical_and(
                        np.logical_and(np.logical_and(original_joints[:, 0] >= 0, original_joints[:, 0] < self.size[0]),
                                       np.logical_and(original_joints[:, 1] >= 0, original_joints[:, 1] < self.size[0])),
                        joints_vis[:, 0])

        return torch.cat([img, fake_depth], 0), norm_joints, original_joints, joints_vis

    def COCOReduce(self, norm_joints, original_joints2d, joints_vis):
        # COCO: 0 - "nose", 1 - "left_eye", 2 - "right_eye", 3 - "left_ear", 4 - "right_ear",
        #       5 - "left_shoulder", 6 - "right_shoulder", 7 - "left_elbow", 8 - "right_elbow",
        #       9 - "left_wrist", 10 - "right_wrist", 11 - "left_hip", 12 - "right_hip",
        #       13 - "left_knee", 14 - "right_knee", 15 - "left_ankle", 16 - "right_ankle"
        # MPII: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
        #       5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
        #       10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder,
        #       14 - l elbow, 15 - l wrist
        # Reduce: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
        #         5 - l ankle, 6 - head top,
        #         7 - r wrist, 8 - r elbow, 9 - r shoulder, 10 - l shoulder,
        #         11 - l elbow, 12 - l wrist
        select_ind = [16, 14, 12, 11, 13, 15, 0, 10, 8, 6, 5, 7, 9]
        return norm_joints[select_ind].reshape(13, 2), original_joints2d[select_ind].reshape(13, 2), joints_vis[select_ind]

    def KinectReduce(self, joints):
        select_ind = [14, 13, 12, 16, 17, 18, 3, 6, 5, 4, 8, 9, 10]
        return joints[select_ind].reshape(13, 2)

    def __getitem__(self, index):
        if index < len(self.db):
            rgbd, norm_joints, original_joints2d, joints_vis = self.coco_getitem(index)
            norm_joints, original_joints2d, joints_vis = self.COCOReduce(norm_joints, original_joints2d, joints_vis)
            joints3d = torch.zeros([self.num_joints, 3])
            true_depth = 0
            depth_mask = torch.zeros_like(rgbd[0, :, :])
        else:
            rgbd, _, joints3d, resize_param, skeleton_dict = NTURGBD3DSkeleton.__getitem__(self, index - len(self.db), return_resize_param=True)
            joints2d = []
            for joint in skeleton_dict['joints'][0]['d_loc']:
                joints2d.append(joint)
            joints2d = np.array(joints2d, dtype=np.float32)
            joints2d = self.KinectReduce(joints2d)

            i, j, h, w, need_flip, original_h, original_w = resize_param

            norm_joints = self.normalize_joints_myway(joints2d)
            if self.random_flip and resize_param[-1]:
                norm_joints = self.flip_normalized_joints(norm_joints)

            joints_vis = np.logical_and(np.logical_and(joints2d[:, 1] > i, joints2d[:, 1] < i + h),
                                        np.logical_and(joints2d[:, 0] > j, joints2d[:, 1] < j + w))
            original_joints2d = joints2d[:, ::-1].copy()
            original_joints2d[:, 0] = (original_joints2d[:, 0] - i) / h * self.size[0]
            original_joints2d[:, 1] = (original_joints2d[:, 1] - j) / w * self.size[0]
            true_depth = 1
            
            depth = rgbd[3, :, :]
            depth_mask = depth > 0

            mean = depth.sum() / depth_mask.sum()
            n = depth_mask.sum()
            std = torch.sqrt((((depth - mean) ** 2) * depth_mask).sum() / (n - 1))
            norm_depth = (depth - mean)
            norm_depth[~depth_mask] = 0
            rgbd[3:, :, :] = norm_depth.unsqueeze(0)

        original_joints2d[np.logical_not(joints_vis), :] = 0
        norm_joints[np.logical_not(joints_vis), :] = 0
        scale = generate_scale_mpii(original_joints2d, joints_vis)
        
        return rgbd, index, torch.from_numpy(norm_joints.copy().astype(np.float32)), \
               joints3d, torch.from_numpy(original_joints2d.copy()), \
               torch.from_numpy(joints_vis.astype(np.int32).copy()), true_depth, \
               depth_mask.float(), scale
            #    i, j, h, w, original_h, original_w

class NTURGBDSegJoint(NTURGBD3D2DSkeleton):
    def __init__(self, ntu_root, ntu_file_list, seg_root, seg_image_set, size=320, random_flip=False, random_resized_crop=False, use_jigsaw=False, need_gt=False,
                 only_seg=False, mask_seg_depth=False, mask_seg_rgb=False):
        super(NTURGBDSegJoint, self).__init__(ntu_root, ntu_file_list, size, random_flip, random_resized_crop, use_jigsaw, need_gt)
        self.seg_root = seg_root
        self.seg_image_set = seg_image_set
        self.mpii_num_joints = 16
        self.only_seg = only_seg
        self.mask_seg_depth = mask_seg_depth
        self.mask_seg_rgb = mask_seg_rgb

        with open(self.seg_image_set, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        lines = sorted(lines)

        def convert2depth(fn):
            fn_list = fn.split('/')
            fn_list[0] = 'depth'
            fn_list[1] = 'MDepth-' + fn_list[1].split('.')[0] + '.png'
            return '/'.join(fn_list)

        def convert2gt(fn):
            fn_list = fn.split('/')
            fn_list[0] = 'png_annotation_v2'
            fn_list[1] = fn_list[1].split('.')[0] + '.png'
            return '/'.join(fn_list)

        compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})F(\d{3}).*')
        def convert2skeleton(fn):
            match = re.match(compiled_regex, fn)
            setup, camera, performer, replication, action, frame = [*map(int, match.groups())]
            tag = fn.split('/')[-1][:-8]
            if setup < 18:
                skeleton_fname = os.path.join('./data/NTURGBD/NTURGBD/', \
                                              'nturgb+d_parsed_skeleton', tag, \
                                              'Skeleton-{:08d}.pkl'.format(frame))
            else:
                skeleton_fname = os.path.join('./data/NTURGBD/NTURGBD120/', \
                                              'nturgb+d_parsed_skeleton', tag, \
                                              'Skeleton-{:08d}.pkl'.format(frame))
            return skeleton_fname

        self.seg_image_list = [os.path.join(self.seg_root, l) for l in lines]
        self.seg_depth_list = [os.path.join(self.seg_root, convert2depth(l)) for l in lines]
        self.seg_skeleton_list = [convert2skeleton(l) for l in lines]
        self.seg_gt_list = [os.path.join(self.seg_root, convert2gt(l)) for l in lines]

        self.split = len(self.image_list)

        if not only_seg:
            self.image_list = self.image_list + self.seg_image_list
            self.depth_list = self.depth_list + self.seg_depth_list
            self.skeleton_list = self.skeleton_list + self.seg_skeleton_list
        else:
            self.image_list = self.seg_image_list
            self.depth_list = self.seg_depth_list
            self.skeleton_list = self.seg_skeleton_list


        self.original_label = np.array([0, 1, 2, 3, 6, 7, 8, 17, 18, 19, 25, 26, 27, 32, 33, 34, 38, 39, 43, 44, 46, 49, 50, 56, 58])
        self.label_mapper = np.arange(60)
        for i, l in enumerate(self.original_label):
            self.label_mapper[l] = i

    def Kinect2MPII(self, joints):
        select_ind = [14, 13, 12, 16, 17, 18, 0, 1, 2, 3, 6, 5, 4, 8, 9, 10]
        return joints[select_ind].reshape(16, 2)

    def __len__(self):
        return len(self.image_list)

    def normalize_joints_myway(self, _joints2d, root_index=6):
        joints2d = _joints2d.copy()
        joints2d -= joints2d[root_index, :]
        joints2d = joints2d[:, ::-1]
        s = max(joints2d.max(), np.abs(joints2d.min()))
        joints2d /= s
        return joints2d

    def __getitem__(self, index):
        rgbd, _, joints3d, resize_param, skeleton_dict = NTURGBD3DSkeleton.__getitem__(self, index, return_resize_param=True)
        joints2d = []
        for joint in skeleton_dict['joints'][0]['d_loc']:
            joints2d.append(joint)
        joints2d = np.array(joints2d, dtype=np.float32)
        joints2d = self.Kinect2MPII(joints2d)
        num_joints = joints2d.shape[0]
        assert num_joints == self.mpii_num_joints

        i, j, h, w, need_flip, original_h, original_w = resize_param

        norm_joints = self.normalize_joints_myway(joints2d)
        if self.random_flip and resize_param[-1]:
            norm_joints = self.flip_normalized_joints(norm_joints)

        joints_vis = np.logical_and(np.logical_and(joints2d[:, 1] > i, joints2d[:, 1] < i + h),
                                    np.logical_and(joints2d[:, 0] > j, joints2d[:, 1] < j + w))
        original_joints2d = joints2d[:, ::-1].copy()
        original_joints2d[:, 0] = (original_joints2d[:, 0] - i) / h * self.size[0]
        original_joints2d[:, 1] = (original_joints2d[:, 1] - j) / w * self.size[0]
        true_depth = 1

        depth = rgbd[3, :, :]
        depth_mask = depth > 0

        if depth_mask.sum() == 0:
            mean = 0.0
        else:
            mean = depth.sum() / depth_mask.sum()
        n = depth_mask.sum()
        std = torch.sqrt((((depth - mean) ** 2) * depth_mask).sum() / (n - 1))
        norm_depth = (depth - mean)
        norm_depth[~depth_mask] = 0
        rgbd[3:, :, :] = norm_depth.unsqueeze(0)

        original_joints2d[np.logical_not(joints_vis), :] = 0
        norm_joints[np.logical_not(joints_vis), :] = 0

        scale = generate_scale_mpii(original_joints2d, joints_vis)

        if index >= self.split or self.only_seg:
            if not self.only_seg:
                gt_fname = self.seg_gt_list[index - self.split]
            else:
                gt_fname = self.seg_gt_list[index]
            label = Image.open(gt_fname)
            label = TF.resized_crop(label, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
            label = self.label_mapper[np.array(label).astype(np.uint8)]
            label = torch.from_numpy(label)
            assert not self.random_flip
            true_label = 1
        else:
            label = torch.zeros_like(rgbd[0], dtype=torch.uint8) + 255
            true_label = 0

        if self.mask_seg_depth:
            if index >= self.split and not self.only_seg:
                true_depth = 0
                depth_mask = torch.zeros_like(rgbd[0, :, :])
                rgbd = torch.cat([rgbd[:3], torch.zeros_like(rgbd[:3])], 0)

        true_rgb = 1
        if self.mask_seg_rgb:
            if index >= self.split and not self.only_seg:
                true_rgb = 0
                rgbd = torch.cat([torch.zeros_like(rgbd[:3]), rgbd[3:]], 0)

        grid_x = torch.arange(original_h)
        grid_y = torch.arange(original_w)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
        grid_x = Image.fromarray(grid_x.numpy().astype(np.uint16))
        grid_y = Image.fromarray(grid_y.numpy().astype(np.uint16))
        grid_x = TF.resized_crop(grid_x, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
        grid_y = TF.resized_crop(grid_y, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
        grid_xy = torch.from_numpy(np.stack([np.array(grid_x), np.array(grid_y)], -1).astype(np.int32))

        return rgbd, index, torch.from_numpy(norm_joints.copy().astype(np.float32)), \
                joints3d, torch.from_numpy(original_joints2d.copy()), \
                torch.from_numpy(joints_vis.astype(np.int32).copy()), true_depth, \
                depth_mask.float(), scale, label, true_label, true_rgb, \
                grid_xy, int(original_h), int(original_w), float(mean)

modal2Dataset = {
    'NTURGBD': NTURGBD,
    'NTURGBDS': NTURGBD3DSkeleton,
    'NTURGBDHM': NTURGBD3D2DSkeleton,
    'NTUMPIIRGBDHM': NTUMPIIRGBD3D2DSkeleton,
    'NTUMPIIRGBD2S': NTUMPIIRGBD3D2DSkeletonGCN,
    'NTUCOCORGBD2S': NTUCOCORGBD3D2DSkeletonGCN,
    'NTUSegRGBD2S': NTURGBDSegJoint,
}
