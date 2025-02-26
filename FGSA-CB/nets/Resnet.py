
# # 代码改编自：
# # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# #
# # BSD 3-Clause License
# #
# # [License text]
#
# import torch
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from adain import AdaptiveInstanceNormalization
# import torch.nn.functional as F
# import numpy as np
# import functools
#
# __all__ = ['ResNet', 'resnet50']
#
# model_urls = {
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
# }
#
# def calc_ins_mean_std(x, eps=1e-5):
#     """计算特征图的均值和标准差。"""
#     size = x.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     var = x.view(N, C, -1).var(dim=2) + eps
#     std = var.sqrt().view(N, C, 1, 1)
#     mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return mean, std
#
# def instance_norm_mix(content_feat, style_feat):
#     """用风格特征的统计量替换内容特征的统计量。"""
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_ins_mean_std(style_feat)
#     content_mean, content_std = calc_ins_mean_std(content_feat)
#
#     normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)
#
# def cn_rand_bbox(size, beta, bbx_thres):
#     """采样一个用于裁剪的边界框。"""
#     W = size[2]
#     H = size[3]
#     while True:
#         ratio = np.random.beta(beta, beta)
#         cut_rat = np.sqrt(ratio)
#         cut_w = int(W * cut_rat)
#         cut_h = int(H * cut_rat)
#
#         # 随机采样中心点
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)
#
#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)
#
#         ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
#         if ratio > bbx_thres:
#             break
#
#     return bbx1, bby1, bbx2, bby2
#
# def cn_op_2ins_space_chan(x, crop='neither', beta=1, bbx_thres=0.1, lam=None, chan=False):
#     """使用裁剪的两实例CrossNorm。"""
#     assert crop in ['neither', 'style', 'content', 'both']
#     ins_idxs = torch.randperm(x.size(0)).to(x.device)
#
#     if crop in ['style', 'both']:
#         bbx3, bby3, bbx4, bby4 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
#         x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
#     else:
#         x2 = x[ins_idxs]
#
#     if chan:
#         chan_idxs = torch.randperm(x.size(1)).to(x.device)
#         x2 = x2[:, chan_idxs, :, :]
#
#     if crop in ['content', 'both']:
#         x_aug = torch.zeros_like(x)
#         bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
#         x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(
#             content_feat=x[:, :, bbx1:bbx2, bby1:bby2], style_feat=x2)
#
#         mask = torch.ones_like(x)
#         mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
#         x_aug = x * mask + x_aug
#     else:
#         x_aug = instance_norm_mix(content_feat=x, style_feat=x2)
#
#     if lam is not None:
#         x = x * lam + x_aug * (1 - lam)
#     else:
#         x = x_aug
#
#     return x
#
# class CrossNorm(nn.Module):
#     """CrossNorm模块。"""
#     def __init__(self, crop=None, beta=None):
#         super(CrossNorm, self).__init__()
#         self.active = False
#         self.cn_op = functools.partial(cn_op_2ins_space_chan, crop=crop, beta=beta)
#
#     def forward(self, x):
#         if self.training and self.active:
#             x = self.cn_op(x)
#         self.active = False
#         return x
#
# class SelfNorm(nn.Module):
#     """SelfNorm模块。"""
#     def __init__(self, chan_num, is_two=False):
#         super(SelfNorm, self).__init__()
#         # 通道级别的全连接层
#         self.g_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2,
#                               bias=False, groups=chan_num)
#         self.g_bn = nn.BatchNorm1d(chan_num)
#
#         if is_two:
#             self.f_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2,
#                                   bias=False, groups=chan_num)
#             self.f_bn = nn.BatchNorm1d(chan_num)
#         else:
#             self.f_fc = None
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         mean, std = calc_ins_mean_std(x, eps=1e-12)
#         statistics = torch.cat((mean.squeeze(3), std.squeeze(3)), -1)
#
#         g_y = self.g_fc(statistics)
#         g_y = self.g_bn(g_y)
#         g_y = torch.sigmoid(g_y)
#         g_y = g_y.view(b, c, 1, 1)
#
#         if self.f_fc is not None:
#             f_y = self.f_fc(statistics)
#             f_y = self.f_bn(f_y)
#             f_y = torch.sigmoid(f_y)
#             f_y = f_y.view(b, c, 1, 1)
#             return x * g_y.expand_as(x) + mean.expand_as(x) * (f_y.expand_as(x) - g_y.expand_as(x))
#         else:
#             return x * g_y.expand_as(x)
#
# class CNSN(nn.Module):
#     """用于组合CrossNorm和SelfNorm的模块。"""
#     def __init__(self, crossnorm, selfnorm):
#         super(CNSN, self).__init__()
#         self.crossnorm = crossnorm
#         self.selfnorm = selfnorm
#
#     def forward(self, x):
#         if self.crossnorm and self.crossnorm.active:
#             x = self.crossnorm(x)
#         if self.selfnorm:
#             x = self.selfnorm(x)
#         return x
#
# class Bottleneck(nn.Module):
#     """
#     ResNet的瓶颈层。
#     """
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  fs=0, use_crossnorm=False, use_selfnorm=False):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.fs = fs
#         if self.fs == 1:
#             self.instance_norm_layer = AdaptiveInstanceNormalization()
#             self.relu = nn.ReLU(inplace=False)
#         else:
#             self.relu = nn.ReLU(inplace=True)
#
#         self.use_crossnorm = use_crossnorm
#         self.use_selfnorm = use_selfnorm
#
#         if self.use_crossnorm:
#             self.crossnorm = CrossNorm()
#         else:
#             self.crossnorm = None
#         if self.use_selfnorm:
#             self.selfnorm = SelfNorm(planes * self.expansion)
#         else:
#             self.selfnorm = None
#
#         if self.use_crossnorm or self.use_selfnorm:
#             self.cnsn = CNSN(self.crossnorm, self.selfnorm)
#         else:
#             self.cnsn = None
#
#     def forward(self, x_tuple):
#         if len(x_tuple) == 1:
#             x = x_tuple[0]
#         elif len(x_tuple) == 3:
#             x, x_t, x_st = x_tuple
#         else:
#             raise NotImplementedError(f"{len(x_tuple)} is not supported length of the tuple")
#
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#
#         # 如果适用，应用CrossNorm和SelfNorm
#         if self.cnsn is not None:
#             if self.training and self.use_crossnorm:
#                 self.crossnorm.active = True
#             out = self.cnsn(out)
#
#         out = self.relu(out)
#
#         if len(x_tuple) == 3:
#             # 对x_t和x_st进行类似处理
#             with torch.no_grad():
#                 residual_t = x_t
#                 out_t = self.conv1(x_t)
#                 out_t = self.bn1(out_t)
#                 out_t = self.relu(out_t)
#                 out_t = self.conv2(out_t)
#                 out_t = self.bn2(out_t)
#                 out_t = self.relu(out_t)
#                 out_t = self.conv3(out_t)
#                 out_t = self.bn3(out_t)
#                 if self.downsample is not None:
#                     residual_t = self.downsample(x_t)
#                 out_t += residual_t
#                 if self.cnsn is not None:
#                     out_t = self.cnsn(out_t)
#                 out_t = self.relu(out_t)
#
#             residual_st = x_st
#             out_st = self.conv1(x_st)
#             out_st = self.bn1(out_st)
#             out_st = self.relu(out_st)
#             out_st = self.conv2(out_st)
#             out_st = self.bn2(out_st)
#             out_st = self.relu(out_st)
#             out_st = self.conv3(out_st)
#             out_st = self.bn3(out_st)
#             if self.downsample is not None:
#                 residual_st = self.downsample(x_st)
#             out_st += residual_st
#             if self.cnsn is not None:
#                 out_st = self.cnsn(out_st)
#             out_st = self.relu(out_st)
#
#             return [out, out_t, out_st]
#         else:
#             return [out]
#
# class ResNet(nn.Module):
#     """
#     ResNet全局模块的初始化。
#     """
#     def __init__(self, block, layers, fs_layer=None, num_classes=1000,
#                  use_crossnorm=False, use_selfnorm=False):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#
#         self.fs_layer = fs_layer if fs_layer is not None else [0]
#
#         if self.fs_layer[0] == 1:
#             self.bn1 = AdaptiveInstanceNormalization()
#             self.relu = nn.ReLU(inplace=False)
#         else:
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace=True)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], fs_layer=self.fs_layer[1],
#                                        use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, fs_layer=self.fs_layer[2],
#                                        use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, fs_layer=self.fs_layer[3],
#                                        use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, fs_layer=self.fs_layer[4],
#                                        use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         # 初始化权重
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
#                 if m.weight is not None:
#                     nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, fs_layer=0,
#                     use_crossnorm=False, use_selfnorm=False):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, fs=0,
#                             use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm))
#         self.inplanes = planes * block.expansion
#         for index in range(1, blocks):
#             layers.append(block(self.inplanes, planes,
#                                 fs=0 if (fs_layer > 0 and index < blocks - 1) else fs_layer,
#                                 use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1([x])[0]
#         x = self.layer2([x])[0]
#         x = self.layer3([x])[0]
#         x = self.layer4([x])[0]
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
# def forgiving_state_restore(net, loaded_dict):
#     """
#     当某些张量的大小不匹配时，处理部分加载。
#     因为我们希望使用在不同类别数量上训练的模型。
#     """
#     net_state_dict = net.state_dict()
#     new_loaded_dict = {}
#     for k in net_state_dict:
#         if (k in loaded_dict) and (net_state_dict[k].size() == loaded_dict[k].size()):
#             new_loaded_dict[k] = loaded_dict[k]
#         else:
#             print(f"Skipped loading parameter {k}")
#     net_state_dict.update(new_loaded_dict)
#     net.load_state_dict(net_state_dict)
#     return net
#
# def resnet50(pretrained=True, fs_layer=[1, 0, 1, 0, 0],
#               use_crossnorm=False, use_selfnorm=False, **kwargs):
#     """构建ResNet-50模型。"""
#     if fs_layer is None:
#         fs_layer = [0, 0, 0, 0, 0]
#     model = ResNet(Bottleneck, [3, 4, 6, 3], fs_layer=fs_layer,
#                    use_crossnorm=use_crossnorm, use_selfnorm=use_selfnorm, **kwargs)
#     if pretrained:
#         print("########### 加载预训练模型 ##############")
#         forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
#     return model
#
#
#










































import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from adain import AdaptiveInstanceNormalization
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
import functools
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# # 定义一个可视化函数，将Tensor转换为PIL格式并显示
# # 修改后的可视化函数，将多通道张量转换为3通道RGB格式以便可视化
# def visualize_tensor(tensor, title="Image after Style Transfer", folder_path="output_images", step=0):
#     # 检查并创建文件夹
#     os.makedirs(folder_path, exist_ok=True)
#
#     # 如果张量的通道数超过3，则仅选择前三个通道
#     if tensor.size(1) > 3:
#         tensor = tensor[:, :3, :, :]
#
#     # 将tensor转换为PIL格式
#     image_tensor = tensor[0].cpu().detach().clone()
#     image_tensor = transforms.ToPILImage()(image_tensor)
#
#     # 保存图像到指定文件夹中
#     filename = os.path.join(folder_path, f"{title}_step_{step}.png")
#     plt.imshow(image_tensor)
#     plt.title(title)
#     plt.axis('off')
#     plt.savefig(filename)
#     plt.close()
#     print(f"Image saved at: {filename}")


__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
############################################################################################################
# 计算实例归一化的均值和标准差
def calc_ins_mean_std(x, eps=1e-5):
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


class SelfNorm(nn.Module):
    """SelfNorm module for fine-tuning the source style"""
    def __init__(self, chan_num, is_two=False):
        super(SelfNorm, self).__init__()

        # Channel-wise fully connected layer for recalibrating statistics
        self.g_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2, bias=False, groups=chan_num)
        self.g_bn = nn.BatchNorm1d(chan_num)

        # Optionally use f_fc for two-channel statistics
        if is_two:
            self.f_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2, bias=False, groups=chan_num)
            self.f_bn = nn.BatchNorm1d(chan_num)
        else:
            self.f_fc = None

    def forward(self, x, x_t=None):
        b, c, _, _ = x.size()

        # Calculate mean and std for x (source domain)
        mean, std = calc_ins_mean_std(x, eps=1e-12)

        statistics = torch.cat((mean.squeeze(3), std.squeeze(3)), -1)

        # Apply the channel-wise fully connected layer and batchnorm
        g_y = self.g_fc(statistics)
        g_y = self.g_bn(g_y)
        g_y = torch.sigmoid(g_y)
        g_y = g_y.view(b, c, 1, 1)

        # If we are using two channels and have target domain statistics
        if self.f_fc is not None and x_t is not None:
            mean_t, std_t = calc_ins_mean_std(x_t, eps=1e-12)
            f_y = self.f_fc(torch.cat((mean_t.squeeze(3), std_t.squeeze(3)), -1))
            f_y = self.f_bn(f_y)
            f_y = torch.sigmoid(f_y)
            f_y = f_y.view(b, c, 1, 1)

            # Adjust the statistics of x (source) based on x_t (target)
            return x * g_y.expand_as(x) + mean.expand_as(x) * (f_y.expand_as(x) - g_y.expand_as(x))
        else:
            return x * g_y.expand_as(x)


class Bottleneck(nn.Module):
    """Bottleneck Layer for ResNet"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, fs=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        #self.selfnorm_layer = SelfNorm(planes * self.expansion, is_two=True)  # SelfNorm using target domain statistics

        # Initialize AdaIN and SelfNorm if needed
        self.fs = fs
        if self.fs == 1:
            self.instance_norm_layer = AdaptiveInstanceNormalization()
            # self.selfnorm_layer = SelfNorm(planes * self.expansion, is_two=True)  # SelfNorm using target domain statistics
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple):
        if len(x_tuple) == 1:
            x = x_tuple[0]
        elif len(x_tuple) == 3:
            x = x_tuple[0]
            x_t = x_tuple[1]  # Target domain
            x_st = x_tuple[2]  # Source domain
        else:
            raise NotImplementedError("%d is not supported length of the tuple" % (len(x_tuple)))

        residual = x

        # First convolutional block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        # Handle x_t and x_st (for domain adaptation)
        if len(x_tuple) == 3:
            with torch.no_grad():
                residual_t = x_t

                out_t = self.conv1(x_t)
                out_t = self.bn1(out_t)
                out_t = self.relu(out_t)

                out_t = self.conv2(out_t)
                out_t = self.bn2(out_t)
                out_t = self.relu(out_t)

                out_t = self.conv3(out_t)
                out_t = self.bn3(out_t)

                if self.downsample is not None:
                    residual_t = self.downsample(x_t)

                out_t += residual_t

            residual_st = x_st

            out_st = self.conv1(x_st)
            out_st = self.bn1(out_st)
            out_st = self.relu(out_st)

            out_st = self.conv2(out_st)
            out_st = self.bn2(out_st)
            out_st = self.relu(out_st)

            out_st = self.conv3(out_st)
            out_st = self.bn3(out_st)

            if self.downsample is not None:
                residual_st = self.downsample(x_st)

            out_st += residual_st

        # Apply AdaIN and then SelfNorm only on out_st using out_t for style matching
        if self.fs == 1:
            out = self.instance_norm_layer(out)

            if len(x_tuple) == 3:
                out_st = self.instance_norm_layer(out_st, out_t) + out_st
                #out_st = self.instance_norm_layer(out_st, out_t)

                with torch.no_grad():
                    out_t = self.instance_norm_layer(out_t)

                # Apply SelfNorm using out_t (target domain) to adjust out_st (source domain)
                #out_st = self.selfnorm_layer(out_st, out_t)
                # # **插入可视化**
                # visualize_tensor(out_st, title="Image after Style Transfer")

        out = self.relu(out)

        if len(x_tuple) == 3:
            with torch.no_grad():
                out_t = self.relu(out_t)
            out_st = self.relu(out_st)
            return [out, out_t, out_st]
        else:
            return [out]

########################################################################



























###########################################################################################
# class Bottleneck(nn.Module):
#     """
#     Bottleneck Layer for Resnet
#     """
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, fs=0):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.fs = fs
#         if self.fs == 1:
#             self.instance_norm_layer = AdaptiveInstanceNormalization()
#             self.relu = nn.ReLU(inplace=False)
#         else:
#             self.relu = nn.ReLU(inplace=True)
#
#
#
#     def forward(self, x_tuple):
#
#
#         if len(x_tuple) == 1:
#             x = x_tuple[0]
#         elif len(x_tuple) == 3:
#             x = x_tuple[0]
#             x_t = x_tuple[1]
#             x_st = x_tuple[2]
#
#         else:
#             raise NotImplementedError("%d is not supported length of the tuple"%(len(x_tuple)))
#
#         residual = x
#
#         out = self.conv1(x)
#
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#
#
#
#         if len(x_tuple) == 3:
#             with torch.no_grad():
#                 residual_t = x_t
#
#                 out_t = self.conv1(x_t)
#
#                 out_t = self.bn1(out_t)
#                 out_t = self.relu(out_t)
#
#                 out_t = self.conv2(out_t)
#                 out_t = self.bn2(out_t)
#                 out_t = self.relu(out_t)
#
#                 out_t = self.conv3(out_t)
#                 out_t = self.bn3(out_t)
#
#                 if self.downsample is not None:
#                     residual_t = self.downsample(x_t)
#
#                 out_t += residual_t
#
#
#
#
#             residual_st = x_st
#
#             out_st = self.conv1(x_st)
#
#             out_st = self.bn1(out_st)
#             out_st = self.relu(out_st)
#
#             out_st = self.conv2(out_st)
#             out_st = self.bn2(out_st)
#             out_st = self.relu(out_st)
#
#             out_st = self.conv3(out_st)
#             out_st = self.bn3(out_st)
#
#             if self.downsample is not None:
#                 residual_st = self.downsample(x_st)
#
#             out_st += residual_st
#
#
#         if self.fs == 1:
#             out = self.instance_norm_layer(out)
#             if len(x_tuple) == 3:
#                 out_st = self.instance_norm_layer(out_st, out_t) +out_st
#                 with torch.no_grad():
#                     out_t = self.instance_norm_layer(out_t)
#
#         out = self.relu(out)
#
#         if len(x_tuple) == 3:
#             with torch.no_grad():
#                 out_t = self.relu(out_t)
#             out_st = self.relu(out_st)
#             return [out, out_t, out_st]
#         else:
#             return [out]



#############################################################################################################################
class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, fs_layer=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.fs_layer = fs_layer if fs_layer is not None else [0]

        if fs_layer[0] == 1:

            self.bn1 = AdaptiveInstanceNormalization()
            self.relu = nn.ReLU(inplace=False)
        else:
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], fs_layer=fs_layer[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, fs_layer=fs_layer[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, fs_layer=fs_layer[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, fs_layer=fs_layer[4])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, fs_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fs=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                fs=0 if (fs_layer > 0 and index < blocks - 1) else fs_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)


        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net



def resnet50(pretrained=True, fs_layer=[1,0, 1, 0, 0], **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if fs_layer is None:
        fs_layer = [0, 0, 0, 0, 0]
    model = ResNet(Bottleneck, [3, 4, 6, 3], fs_layer=fs_layer, **kwargs)
    if pretrained:
        print("########### pretrained ##############")
        forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))


    return model