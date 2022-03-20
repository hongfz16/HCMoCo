from __future__ import print_function

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage import color
from torchvision import transforms, datasets

from .dataset import ImageFolderInstance
from .RandAugment import rand_augment_transform

import torch.nn as nn

class StackTransform(object):
    """transform a group of images independently"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        return torch.stack([self.transform(crop) for crop in imgs])


class JigsawCrop(object):
    """Jigsaw style crop"""
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        crops = [Image.fromarray(crop) for crop in crops]
        return crops


class Rotate(object):
    """rotation"""
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = np.random.choice(self.angles)
        if isinstance(img, Image.Image):
            img = img.rotate(angle, fillcolor=(128, 128, 128))
            return img
        elif isinstance(img, np.ndarray):
            if angle == 0:
                pass
            elif angle == 90:
                img = np.flipud(np.transpose(img, (1, 0, 2)))
            elif angle == 180:
                img = np.fliplr(np.flipud(img))
            elif angle == 270:
                img = np.transpose(np.flipud(img), (1, 0, 2))
            else:
                img = Image.fromarray(img)
                img = img.rotate(angle, fillcolor=(128, 128, 128))
                img = np.asarray(img)
            return img
        else:
            raise TypeError('not supported type in rotation: ', type(img))


class RGB2RGB(object):
    """Dummy RGB transfer."""
    def __call__(self, img):
        return img


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class GaussianBlur2(object):
    """Gaussian Blur version 2"""
    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianBlurBatch(object):
    """blur a batch of images on CPU or GPU"""
    def __init__(self, kernel_size, use_cuda=False, p=0.5):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        assert 0 <= p <= 1.0, 'p is out of range [0, 1]'
        self.p = p
        self.use_cuda = use_cuda
        if use_cuda:
            self.blur = self.blur.cuda()

    def __call__(self, imgs):

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        if self.use_cuda:
            x = x.cuda()

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        bsz = imgs.shape[0]
        n_blur = int(bsz * self.p)
        with torch.no_grad():
            if n_blur == bsz:
                imgs = self.blur(imgs)
            elif n_blur == 0:
                pass
            else:
                imgs_1, imgs_2 = torch.split(imgs, [n_blur, bsz - n_blur], dim=0)
                imgs_1 = self.blur(imgs_1)
                imgs = torch.cat([imgs_1, imgs_2], dim=0)

        return imgs


def build_transforms(aug, modal, use_memory_bank=True):
    if use_memory_bank:
        # memory bank likes 0.08
        crop = 0.08
    else:
        # moco cache likes 0.2
        crop = 0.2

    if modal == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_transfer = RGB2RGB()
    else:
        mean = [0.457, -0.082, -0.052]
        std = [0.500, 1.331, 1.333]
        color_transfer = RGB2YDbDr()
    normalize = transforms.Normalize(mean=mean, std=std)

    if aug == 'A':
        # used in InsDis, MoCo, PIRL
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == 'B':
        # used in MoCoV2
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur2()], p=0.5),
            color_transfer,
            transforms.ToTensor(),
            normalize
        ])
    elif aug == 'C':
        # used in CMC, CMCPIRL
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == 'D':
        # used in InfoMin
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(modal == 'CMC')),
            transforms.RandomGrayscale(p=0.2),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == 'E':
        # used in CMCv2
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(modal == 'CMC')),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('transform not found: {}'.format(aug))

    jigsaw_transform = transforms.Compose([
        transforms.RandomResizedCrop(255, scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(),
        JigsawCrop(),
        StackTransform(transforms.Compose([
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ]))
    ])

    return train_transform, jigsaw_transform


# deprecated
def build_contrast_loader(opt, ngpus_per_node):
    """build loaders for contrastive training"""
    data_folder = opt.data_folder
    aug = opt.aug
    modal = opt.modal
    use_jigsaw = opt.jigsaw
    use_memory_bank = (opt.mem == 'bank')
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    train_transform, jigsaw_transform = \
        build_transforms(aug, modal, use_memory_bank)

    train_dir = os.path.join(data_folder, 'train')
    if use_jigsaw:
        train_dataset = ImageFolderInstance(
            train_dir, transform=train_transform,
            two_crop=(not use_memory_bank),
            jigsaw_transform=jigsaw_transform
        )
    else:
        train_dataset = ImageFolderInstance(
            train_dir, transform=train_transform,
            two_crop=(not use_memory_bank)
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    print('train images: {}'.format(len(train_dataset)))

    return train_dataset, train_loader, train_sampler

# deprecated
def build_linear_loader(opt, ngpus_per_node):
    """build loaders for linear evaluation"""
    # transform
    if opt.modal == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_transfer = RGB2RGB()
    else:
        mean = [0.457, -0.082, -0.052]
        std = [0.500, 1.331, 1.333]
        color_transfer = RGB2YDbDr()
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.aug_linear == 'NULL':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_linear == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params,
                                   use_cmc=(opt.modal == 'CMC')),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

    # dataset
    data_folder = opt.data_folder
    train_dir = os.path.join(data_folder, 'train')
    val_dir = os.path.join(data_folder, 'val')
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            color_transfer,
            transforms.ToTensor(),
            normalize,
        ])
    )

    # loader
    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=32, pin_memory=True)

    print('train images: {}'.format(len(train_dataset)))
    print('test images: {}'.format(len(val_dataset)))

    return train_loader, val_loader, train_sampler

from .dataset import modal2Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
from typing import Optional
from torch.utils.data import Dataset, Sampler
from operator import itemgetter

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

def build_own_contrast_loader(opt, ngpus_per_node, need_gt=False, need_val=False):
    modal = opt.modal
    assert opt.dataset + modal in modal2Dataset.keys()
    use_jigsaw = opt.jigsaw
    assert not use_jigsaw
    assert not need_gt

    random_flip = True if opt.random_flip else False

    batch_size = int(opt.batch_size / opt.world_size)
    num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    assert not opt.downstream_training
    if opt.dataset in ['NTU', '']:
        train_dataset = modal2Dataset[opt.dataset + modal](opt.data_folder, opt.train_file_list, random_flip=random_flip, random_resized_crop=True, need_gt=need_gt, need_geodesic=need_geodesic)
    else:
        if opt.dataset + modal == 'NTUMPIIRGBD2S':
            train_dataset = modal2Dataset['NTUMPIIRGBD2S'](opt.data_folder, opt.train_file_list, opt.mpii_root, 'train', random_flip=random_flip,
                                                        random_resized_crop=True, need_gt=need_gt)
        elif opt.dataset + modal == 'NTUCOCORGBD2S':
            train_dataset = modal2Dataset['NTUCOCORGBD2S'](opt.data_folder, opt.train_file_list, opt.coco_root, 'train2014', random_flip=random_flip,
                                                        random_resized_crop=True, need_gt=need_gt)
        elif opt.dataset + modal == 'NTUSegRGBD2S':
            train_dataset = modal2Dataset['NTUSegRGBD2S'](opt.data_folder, opt.train_file_list, opt.seg_root, opt.seg_file_list, random_flip=random_flip,
                                                        random_resized_crop=True, need_gt=need_gt,
                                                        mask_seg_depth=opt.mask_seg_depth, mask_seg_rgb=opt.mask_seg_rgb)
        else:
            train_dataset = modal2Dataset[opt.dataset + modal](opt.data_folder, opt.train_file_list, opt.mpii_root, 'train', random_flip=random_flip, random_resized_crop=True, need_gt=need_gt, need_geodesic=need_geodesic)

    weights = np.zeros([len(train_dataset)])
    try:
        db_len = len(train_dataset.db)
        ntu_len = len(train_dataset.image_list)
    except:
        print("Not using db...")
        if not opt.not_use_weighted_sampler:
            db_len = len(train_dataset) - train_dataset.split
            ntu_len = train_dataset.split
        else:
            db_len = len(train_dataset)
            ntu_len = len(train_dataset)

    if opt.dataset + modal == 'NTUSegRGBD2S':
        weights[:ntu_len] = db_len / len(train_dataset)
        weights[ntu_len:] = ntu_len / len(train_dataset)
    else:
        weights[:db_len] = ntu_len / len(train_dataset)
        weights[db_len:] = db_len / len(train_dataset)
    sampler = WeightedRandomSampler(weights, len(weights))
    train_sampler = DistributedSamplerWrapper(sampler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    if not need_val:
        return train_dataset, train_loader, train_sampler
    else:
        if opt.dataset + modal == 'NTUSegRGBD2S':
            val_dataset = modal2Dataset['NTUSegRGBD2S'](opt.data_folder, opt.train_file_list, opt.seg_root, opt.seg_val_file_list, random_flip=random_flip,
                                                        random_resized_crop=True, need_gt=need_gt, need_geodesic=need_geodesic,
                                                        only_seg=True)
        else:
            val_dataset = modal2Dataset[opt.dataset + modal](opt.data_folder, opt.val_file_list, random_flip=random_flip, random_resized_crop=False, need_gt=need_gt, need_geodesic=need_geodesic)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=(val_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=val_sampler)
        return train_loader, train_dataset, val_loader, train_sampler

