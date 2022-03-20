import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
from hrnet.official_hrnet import get_hrnet_w18_backbone, get_hrnet_w32_backbone, get_hrnet_w48_backbone

class DepthRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes*2, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
    
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNetBackBone(nn.Module):
    def __init__(self):
        super(ResNetBackBone, self).__init__()
        
        modelPreTrain50 = resnet.resnet50(pretrained=True)
        self.model = modelPreTrain50
        
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
              
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        
        return x3,x4  


class A2J_model(nn.Module):
    def __init__(self, num_classes, is_3D=True):
        super(A2J_model, self).__init__()
        self.is_3D = is_3D 
        self.Backbone = ResNetBackBone() # 1 channel depth only, resnet50 
        self.regressionModel = RegressionModel(2048, num_classes=num_classes)
        self.classificationModel = ClassificationModel(1024, num_classes=num_classes)
        if is_3D:
            self.DepthRegressionModel = DepthRegressionModel(2048, num_classes=num_classes)        
    
    def forward(self, x): 
        x3,x4 = self.Backbone(x)
        classification = self.classificationModel(x3)
        regression = self.regressionModel(x4)
        if self.is_3D:
            DepthRegressionModel  = self.DepthRegressionModel(x4)
            return (classification, regression, DepthRegressionModel)
        return (classification, regression)

class A2J_HRNet_model(nn.Module):
    def __init__(self, num_classes, num_anchors, width=18, pretrain_pth=None, is_3D=True):
        super(A2J_HRNet_model, self).__init__()
        self.is_3D = is_3D
        self.width = width
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        if width == 18:
            self.Backbone = get_hrnet_w18_backbone(pretrain_pth)
            out_channel = sum([18, 36, 72, 144])
        elif width == 32:
            self.Backbone = get_hrnet_w32_backbone(pretrain_pth)
            out_channel = sum([32, 64, 128, 256])
        elif width == 48:
            self.Backbone = get_hrnet_w48_backbone(pretrain_pth)
            out_channel = sum([48, 96, 192, 384])
        else:
            raise NotImplementedError
        
        self.regressionModel = RegressionModel(out_channel, num_anchors=num_anchors, num_classes=num_classes)
        self.classificationModel = ClassificationModel(out_channel, num_anchors=num_anchors, num_classes=num_classes)
        if is_3D:
            self.DepthRegressionModel = DepthRegressionModel(out_channel, num_anchors=num_anchors, num_classes=num_classes)

    def merge_all_res(self, x):
        ALIGN_CORNERS=False
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x1, x2, x3], 1)
        return x

    def forward(self, x):
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
        
        feat_list = self.Backbone(x)
        merged_feat = self.merge_all_res(feat_list)
        classification = self.classificationModel(merged_feat)
        regression = self.regressionModel(merged_feat)
        if self.is_3D:
            DepthRegressionModel  = self.DepthRegressionModel(merged_feat)
            return (classification, regression, DepthRegressionModel)
        return (classification, regression)
