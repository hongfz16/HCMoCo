import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, norm_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto'):
        super(ConvModule, self).__init__()
        
        norm = nn.BatchNorm2d(out_channels)
        self.add_module('norm_name', norm)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.act = nn.ReLU()
    
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, 'norm_name')

    def forward(self, x):
        output = self.conv(x)
        output = self.norm1(output)
        output = self.act(output)
        return output

class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, norm_func=nn.BatchNorm2d, num_convs=2, kernel_size=3, dilation=1):
        super(FCNHead, self).__init__()
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.channels = channels
        self.norm_func = norm_func
        self.num_classes = num_classes

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        # convs.append(
        #     nn.Conv2d(
        #         self.in_channels,
        #         self.channels,
        #         self.kernel_size,
        #         stride=1,
        #         padding=conv_padding,
        #         dilation=self.dilation,
        #         groups=1,
        #         bias='auto'
        # ))
        # convs.append(norm_func(self.channels))
        # convs.append(nn.ReLU())
        convs.append(
            ConvModule(
                norm_func,
                self.in_channels,
                self.channels,
                self.kernel_size,
                stride=1,
                padding=conv_padding,
                dilation=self.dilation,
                groups=1,
                bias='auto'
            )
        )
        for i in range(self.num_convs - 1):
            # convs.append(
            #     nn.Conv2d(
            #         self.channels,
            #         self.channels,
            #         self.kernel_size,
            #         stride=1,
            #         padding=conv_padding,
            #         dilation=self.dilation,
            #         groups=1,
            #         bias='auto'
            # ))
            # convs.append(norm_func(self.channels))
            # convs.append(nn.ReLU())
            convs.append(
                ConvModule(
                    norm_func,
                    self.channels,
                    self.channels,
                    self.kernel_size,
                    stride=1,
                    padding=conv_padding,
                    dilation=self.dilation,
                    groups=1,
                    bias='auto'
                )
            )
        self.convs = nn.Sequential(*convs)
        # self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, input):
        output = self.convs(input)
        # output = self.dropout(output)
        logits = self.conv_seg(output)
        w, h = logits.shape[2] * 4, logits.shape[3] * 4
        logits_resize = F.interpolate(logits, size=(w, h), mode='bilinear', align_corners=False)
        return logits_resize
