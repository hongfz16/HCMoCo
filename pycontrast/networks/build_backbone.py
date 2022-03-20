from pickle import NONE
import torch
import numpy as np
import torch.nn as nn
from .resnet import model_dict as resnet_model_dict
from .util import Normalize, JigsawHead
from .official_hrnet.official_hrnet import get_hrnet_w48_backbone, get_hrnet_w32_backbone, get_hrnet_w18_backbone
import torch.nn.functional as F
from .SGCN.create_SGCN import create_sgcn, create_gcn_mapper
from .pointnet2_msg import Pointnet2MSG
from .pointnet2 import pytorch_utils as pt_utils
from .pointnet2 import pointnet2_utils

class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = resnet_model_dict[name](width=width)

        if head == 'linear':
            self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        feat = self.encoder(x)
        if mode == 0 or mode == 1:
            feat = self.head(feat)
        return feat

class RGBMultiHeads(RGBSingleHead):
    """RGB model with Multiple linear/mlp projection heads"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBMultiHeads, self).__init__(name, head, feat_dim)

        self.head_jig = JigsawHead(dim_in=int(2048*self.width),
                                   dim_out=feat_dim,
                                   head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        if mode == 0:
            feat = self.head(self.encoder(x))
            feat_jig = self.head_jig(self.encoder(x_jig))
            return feat, feat_jig
        elif mode == 1:
            feat = self.head(self.encoder(x))
            return feat
        else:
            feat = self.encoder(x)
            return feat

class CMCSingleHead(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128, in_channel_list=[1, 2], linear_feat_map=False, width=50):
        super(CMCSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width
        self.in_channel_list = in_channel_list

        self.encoder1 = resnet_model_dict[name](width=width, in_channel=in_channel_list[0])
        self.encoder2 = resnet_model_dict[name](width=width, in_channel=in_channel_list[1])

        if head == 'linear':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 2
        elif name.endswith('x2'):
            return name[:-2], 1
        else:
            return name, 0.5

    def forward(self, x, mode=0, return_fm=False):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, self.in_channel_list, dim=1)
        feat1 = self.encoder1(x1, return_fm=return_fm)
        feat2 = self.encoder2(x2, return_fm=return_fm)
        if return_fm:
            return feat1, feat2
        if mode == 0 or mode == 1:
            feat1 = self.head1(feat1)
            feat2 = self.head2(feat2)
        return torch.cat((feat1, feat2), dim=1)

class CMCMultiHeads(CMCSingleHead):
    """CMC model with Multiple linear/mlp projection heads"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128, in_channel_list=[1,2]):
        super(CMCMultiHeads, self).__init__(name, head, feat_dim, in_channel_list)

        self.head1_jig = JigsawHead(dim_in=int(2048*self.width),
                                    dim_out=feat_dim,
                                    head=head)
        self.head2_jig = JigsawHead(dim_in=int(2048*self.width),
                                    dim_out=feat_dim,
                                    head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, self.in_channel_list, dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)

        if mode == 0:
            x1_jig, x2_jig = torch.split(x_jig, self.in_channel_list, dim=1)
            feat1_jig = self.encoder1(x1_jig)
            feat2_jig = self.encoder2(x2_jig)

            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            feat1_jig = self.head1_jig(feat1_jig)
            feat2_jig = self.head2_jig(feat2_jig)
            feat = torch.cat((feat1, feat2), dim=1)
            feat_jig = torch.cat((feat1_jig, feat2_jig), dim=1)
            return feat, feat_jig
        elif mode == 1:
            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            return torch.cat((feat1, feat2), dim=1)
        else:
            return torch.cat((feat1, feat2), dim=1)

class CMC3HRNetSGCNSingleHead(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, name='HRNet', head='linear', feat_dim=128, in_channel_list=[3, 3, 3],
                 linear_feat_map=False, width=18, pool_method='mean', opt=None):
        super(CMC3HRNetSGCNSingleHead, self).__init__()

        self.opt = opt

        assert name == 'HRNet'
        self.in_channel_list = in_channel_list
        self.linear_feat_map = linear_feat_map
        self.width = width
        assert pool_method in ['mean', 'max']
        self.pool_method = pool_method

        if width == 18:
            dim_in = sum([18, 36, 72, 144])
            self.encoder1 = get_hrnet_w18_backbone()
            self.encoder2 = get_hrnet_w18_backbone()
            # assert not self.linear_feat_map

        elif width == 32:
            dim_in = sum([32, 64, 128, 256])
            self.encoder1 = get_hrnet_w32_backbone()
            self.encoder2 = get_hrnet_w32_backbone()
            # assert not self.linear_feat_map

        elif width == 48:
            dim_in = sum([48, 96, 192, 384])
            self.encoder1 = get_hrnet_w48_backbone()
            self.encoder2 = get_hrnet_w48_backbone()
            # assert not self.linear_feat_map
        else:
            raise NotImplementedError

        sgcn_dim = 128
        self.encoder3 = create_sgcn(opt.skeleton_meta_name, sgcn_dim, 4)

        if head == 'linear':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head3 = nn.Sequential(
                nn.Linear(sgcn_dim, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            raise NotImplementedError
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        if self.linear_feat_map:
            self.encoder1_linear = nn.Conv2d(dim_in, sgcn_dim, kernel_size=1, stride=1, bias=True)
            self.encoder2_linear = nn.Conv2d(dim_in, sgcn_dim, kernel_size=1, stride=1, bias=True)

    def merge_all_res(self, x):
        ALIGN_CORNERS=False
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x1, x2, x3], 1)
        return x

    def forward(self, x, s, mode=0, return_fm=False):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, self.in_channel_list, dim=1)
        _feat1 = self.encoder1(x1)
        _feat2 = self.encoder2(x2)
        _feat3 = self.encoder3(s)
        avg_feat1 = []
        avg_feat2 = []
        if self.pool_method == 'mean':
            for i in range(4):
                avg_feat1.append(torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(_feat1[i]), 1))
                avg_feat2.append(torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(_feat2[i]), 1))
        elif self.pool_method == 'max':
            for i in range(4):
                avg_feat1.append(torch.flatten(nn.AdaptiveMaxPool2d((1, 1))(_feat1[i]), 1))
                avg_feat2.append(torch.flatten(nn.AdaptiveMaxPool2d((1, 1))(_feat2[i]), 1))
        else:
            raise NotImplementedError
        avg_feat1 = torch.cat(avg_feat1, 1)
        avg_feat2 = torch.cat(avg_feat2, 1)
        avg_feat3 = _feat3.mean(1)

        if mode == 0 or mode == 1:
            feat1 = self.head1(avg_feat1)
            feat2 = self.head2(avg_feat2)
            feat3 = self.head3(avg_feat3)
        else:
            feat1 = avg_feat1
            feat2 = avg_feat2
            feat3 = avg_feat3
        if return_fm:
            if self.linear_feat_map:
                merge1 = self.merge_all_res(_feat1)
                merge2 = self.merge_all_res(_feat2)
                linear_merge1 = self.encoder1_linear(merge1)
                linear_merge2 = self.encoder2_linear(merge2)
                return _feat1, _feat2, _feat3, torch.cat((feat1, feat2, feat3), dim=1), {
                    'merge1': merge1,
                    'merge2': merge2,
                    'linear_merge1': linear_merge1,
                    'linear_merge2': linear_merge2
                }
            else:
                return _feat1, _feat2, _feat3, avg_feat1, avg_feat2, avg_feat3, torch.cat((feat1, feat2, feat3), dim=1)
        return torch.cat((feat1, feat2, feat3), dim=1)

class CMC3HRNetSGCNPN2SingleHead(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, name='HRNetPN', head='linear', feat_dim=128, in_channel_list=[3, 3, 3],
                 linear_feat_map=False, width=18, pool_method='mean', opt=None):
        super(CMC3HRNetSGCNPN2SingleHead, self).__init__()

        self.opt = opt

        assert name == 'HRNetPN'
        self.in_channel_list = in_channel_list
        self.linear_feat_map = linear_feat_map
        self.width = width
        assert pool_method in ['mean', 'max']
        self.pool_method = pool_method

        if width == 18:
            dim_in = sum([18, 36, 72, 144])
            self.encoder1 = get_hrnet_w18_backbone()
            # self.encoder2 = get_hrnet_w18_backbone()
            # assert not self.linear_feat_map

        elif width == 32:
            dim_in = sum([32, 64, 128, 256])
            self.encoder1 = get_hrnet_w32_backbone()
            # self.encoder2 = get_hrnet_w32_backbone()
            # assert not self.linear_feat_map

        elif width == 48:
            dim_in = sum([48, 96, 192, 384])
            self.encoder1 = get_hrnet_w48_backbone()
            # self.encoder2 = get_hrnet_w48_backbone()
            # assert not self.linear_feat_map
        else:
            raise NotImplementedError

        self.encoder2 = Pointnet2MSG(input_channels=0)
        self.pn_dim = 128

        sgcn_dim = 128
        self.encoder3 = create_sgcn(opt.skeleton_meta_name, sgcn_dim, 4)

        if head == 'linear':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(self.pn_dim, feat_dim),
                Normalize(2)
            )
            self.head3 = nn.Sequential(
                nn.Linear(sgcn_dim, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            raise NotImplementedError
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        if self.linear_feat_map:
            self.encoder1_linear = nn.Conv2d(dim_in, sgcn_dim, kernel_size=1, stride=1, bias=True)
            # self.encoder2_linear = nn.Conv2d(dim_in, sgcn_dim, kernel_size=1, stride=1, bias=True)
            self.encoder2_linear = pt_utils.Conv1d(self.pn_dim, sgcn_dim, bn=True)

    def merge_all_res(self, x):
        ALIGN_CORNERS=False
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x1, x2, x3], 1)
        return x

    def depth2pts(self, depth, depth_mask, grid_xy, ori_h, ori_w, mean):
        # 320 x 320
        h = w = size = depth.shape[-1]
        bs = depth.shape[0]
        
        # wrong
        # depth_min = depth.min().item() - 1
        # x = torch.arange(size).cuda()
        # grid_x, grid_y = torch.meshgrid(x, x)
        # pcd = torch.cat([grid_x.reshape(1, 1, size, size).repeat(bs, 1, 1, 1),
        #                  grid_y.reshape(1, 1, size, size).repeat(bs, 1, 1, 1),
        #                  depth[:, 0, :, :].unsqueeze(1)], 1)
        # worldX = (pcd[:, 0, :, :] - 160) * pcd[:, 2, :, :] * 0.0035
        # worldY = (160 - pcd[:, 1, :, :]) * pcd[:, 2, :, :] * 0.0035
        # worldZ = pcd[:, 2, :, :]

        # correct
        mean = mean.reshape(bs, 1, 1, 1)
        pcd = torch.cat([grid_xy[:, :, :, 0].reshape(bs, 1, size, size).float(),
                         grid_xy[:, :, :, 1].reshape(bs, 1, size, size).float(),
                         depth[:, 0, :, :].unsqueeze(1) + mean], 1)
        worldX = (pcd[:, 0, :, :] - ori_h/2) * pcd[:, 2, :, :] * 0.0035
        worldY = (ori_w/2 - pcd[:, 1, :, :]) * pcd[:, 2, :, :] * 0.0035
        worldZ = pcd[:, 2, :, :] - mean.squeeze(1)

        x = worldX.reshape(bs, 1, h * w).float()
        y = worldY.reshape(bs, 1, h * w).float()
        z = worldZ.reshape(bs, 1, h * w).float()

        original_x = torch.zeros_like(x).cuda()
        original_y = torch.zeros_like(y).cuda()
        original_z = torch.zeros_like(z).cuda()

        original_sampled_x = torch.zeros([bs, 1, 4096]).cuda().float()
        original_sampled_y = torch.zeros([bs, 1, 4096]).cuda().float()
        original_sampled_z = torch.zeros([bs, 1, 4096]).cuda().float()

        valid_depth_prob = F.interpolate(depth_mask.unsqueeze(1).float(), size=(h,w), mode='nearest')
        valid_depth_prob = valid_depth_prob.reshape(bs, h * w)

        valid_depth_prob_sum = valid_depth_prob.sum(-1)
        mask = valid_depth_prob_sum > 0
        valid_depth_prob = valid_depth_prob[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        # bs = mask.sum()

        random_sample_ind = valid_depth_prob.multinomial(num_samples = 4096, replacement=True)
        random_sample_ind = random_sample_ind.unsqueeze(1)

        assert random_sample_ind.shape[-1] == 4096
        assert torch.all(random_sample_ind >= 0) and torch.all(random_sample_ind < h * w)

        sample_x = torch.gather(x, 2, random_sample_ind)
        sample_y = torch.gather(y, 2, random_sample_ind)
        sample_z = torch.gather(z, 2, random_sample_ind)

        original_x[mask] = x
        original_y[mask] = y
        original_z[mask] = z

        original_sampled_x[mask] = sample_x
        original_sampled_y[mask] = sample_y
        original_sampled_z[mask] = sample_z

        return torch.cat([original_sampled_x, original_sampled_y, original_sampled_z], 1), torch.cat([original_x, original_y, original_z], 1), random_sample_ind
        # return torch.cat([sample_x, sample_y, sample_z], 1), torch.cat([x, y, z], 1), random_sample_ind

    def pts2depth(self, sampled_pts, pts, feat, h, w):
        bs, fdim = feat.shape[0], feat.shape[1]
        dist, idx = pointnet2_utils.three_nn(pts.transpose(1, 2).contiguous(), sampled_pts.transpose(1, 2).contiguous())
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(feat.contiguous(), idx, weight)
        return interpolated_feats.reshape(bs, fdim, h, w)

    def forward(self, x, s, depth_mask, grid_xy, original_h, original_w, mean, mode=0, return_fm=False):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, self.in_channel_list, dim=1)
        _feat1 = self.encoder1(x1)

        h, w = x1.shape[-2], x1.shape[-1]
        sample_x2_pn, x2_pn, _ = self.depth2pts(x2, depth_mask, grid_xy, original_h, original_w, mean)
        _feat2 = self.encoder2(sample_x2_pn.transpose(1, 2)) # bs x fdim x 4096

        # np_pn = sample_x2_pn.transpose(1,2).detach().cpu().numpy()
        # np.save('sample_pn.pth', np_pn)
        # exit(0)

        _feat3 = self.encoder3(s)
        avg_feat1 = []
        if self.pool_method == 'mean':
            for i in range(4):
                avg_feat1.append(torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(_feat1[i]), 1))
        elif self.pool_method == 'max':
            for i in range(4):
                avg_feat1.append(torch.flatten(nn.AdaptiveMaxPool2d((1, 1))(_feat1[i]), 1))
        else:
            raise NotImplementedError

        avg_feat1 = torch.cat(avg_feat1, 1)
        avg_feat2 = _feat2.mean(-1)
        avg_feat3 = _feat3.mean(1)

        if mode == 0 or mode == 1:
            feat1 = self.head1(avg_feat1)
            feat2 = self.head2(avg_feat2)
            feat3 = self.head3(avg_feat3)
        else:
            feat1 = avg_feat1
            feat2 = avg_feat2
            feat3 = avg_feat3
        if return_fm:
            if self.linear_feat_map:
                merge1 = self.merge_all_res(_feat1)
                linear_merge1 = self.encoder1_linear(merge1)

                merge2 = _feat2
                linear_merge2 = self.encoder2_linear(_feat2)
                linear_merge2 = self.pts2depth(sample_x2_pn, x2_pn, linear_merge2, h, w)
                linear_merge2 = F.interpolate(linear_merge2, size=(linear_merge1.shape[-2], linear_merge1.shape[-1]))

                return _feat1, _feat2, _feat3, torch.cat((feat1, feat2, feat3), dim=1), {
                    'merge1': merge1,
                    'merge2': merge2,
                    'linear_merge1': linear_merge1,
                    'linear_merge2': linear_merge2
                }
            else:
                return _feat1, _feat2, _feat3, avg_feat1, avg_feat2, avg_feat3, torch.cat((feat1, feat2, feat3), dim=1)
        return torch.cat((feat1, feat2, feat3), dim=1)

NAME_TO_FUNC = {
    'RGBSin': RGBSingleHead,
    'RGBMul': RGBMultiHeads,
    'CMCSin': CMCSingleHead,
    'CMCMul': CMCMultiHeads,
    'RGBD2SHRNetSin': CMC3HRNetSGCNSingleHead,
    'RGBD2SHRNetPNSin': CMC3HRNetSGCNPN2SingleHead,
}

def build_model(opt):
    # specify modal key
    branch = 'Mul' if opt.jigsaw else 'Sin'
    model_key = opt.modal + opt.arch + branch

    model = NAME_TO_FUNC[model_key](opt.arch, opt.head, opt.feat_dim, opt.in_channel_list, opt.linear_feat_map, opt.width, opt.pool_method, opt)
    if opt.IN_Pretrain is not None:
        print("Init Encoder1 from {}".format(opt.IN_Pretrain))
        if opt.arch.startswith('HRNet'):
            pretrained_ckpt = torch.load(opt.IN_Pretrain, map_location='cpu')
            model_state_dict = model.encoder1.state_dict()
            update_model_state = {}
            for k, v in pretrained_ckpt.items():
                if k in model_state_dict.keys():
                    update_model_state[k] = v
                else:
                    print("{} not matched.".format(k))
            model_state_dict.update(update_model_state)
            model.encoder1.load_state_dict(model_state_dict)
        else:
            raise NotImplementedError
    if opt.depth_Pretrain is not None:
        print("Init Encoder2 from {}".format(opt.depth_Pretrain))
        if opt.arch.startswith('HRNet'):
            pretrained_ckpt = torch.load(opt.depth_Pretrain, map_location='cpu')
            model_state_dict = model.encoder2.state_dict()
            update_model_state = {}
            for k, v in pretrained_ckpt.items():
                if k in model_state_dict.keys():
                    update_model_state[k] = v
                else:
                    print("{} not matched.".format(k))
            model_state_dict.update(update_model_state)
            model.encoder2.load_state_dict(model_state_dict)
        else:
            raise NotImplementedError
    if opt.mem == 'moco':
        model_ema = NAME_TO_FUNC[model_key](opt.arch, opt.head, opt.feat_dim)
    else:
        model_ema = None

    return model, model_ema
