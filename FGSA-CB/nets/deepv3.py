"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from Resnet import resnet50
from mynn import initialize_weights, Upsample
from deeplabv3_training import Dice_loss
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from content_loss import get_content_extension_loss, small_get_content_extension_loss
from deeplabv3_training import CE_Loss
import numpy as np
from tllib.self_training .cc_loss import CCConsistency
import torchvision.transforms as transforms

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels * 3, out_channels)

    def forward(self, x1, x2, x4, last=False):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused


SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out





# class ClusteringLoss(nn.Module):
#     def __init__(self, temperature=0.1, epsilon=1e-6):
#         super(ClusteringLoss, self).__init__()
#         self.temperature = temperature
#         self.epsilon = epsilon
#
#     def forward(self, features_source, features_target):
#         # 归一化特征
#         features_source = torch.nn.functional.normalize(features_source, p=2, dim=1)
#         features_target = torch.nn.functional.normalize(features_target, p=2, dim=1)
#
#         # 确保特征张量是二维的
#         features_source = features_source.reshape(features_source.size(0), -1)
#         features_target = features_target.reshape(features_target.size(0), -1)
#
#         # 计算源特征和目标特征之间的相似度矩阵
#         similarities = torch.exp(torch.matmul(features_source, features_target.T) / self.temperature)
#
#         # 限制相似度值以增加数值稳定性
#         similarities = torch.clamp(similarities, max=1.0)
#
#         # 计算每个样本与其他样本的相似度之和
#         sum_similarities = torch.sum(similarities, dim=1)
#
#         # 计算聚类损失
#         loss = -torch.mean(torch.log(sum_similarities + self.epsilon))
#         return loss

class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=Dice_loss, criterion_aux=SoftCrossEntropy_fn, cont_proj_head=128, wild_cont_dict_size=266752,
                variant='D16', skip='m1', skip_num=48, fs_layer=[1,0, 1, 0, 0], use_cel=False, use_sel=False, use_scr=False,cc_consistency_temperature=0.45, cc_consistency_thr=0.7):                         #默认use_cel=False, use_sel=False, use_scr=False
        super(DeepV3Plus, self).__init__()

        # # 初始化 CCConsistency 损失函数
        self.cc_consistency_loss = CCConsistency(temperature=cc_consistency_temperature, thr=cc_consistency_thr)       # 修改


        self.fs_layer = fs_layer
        self.use_cel = use_cel
        self.use_sel = use_sel
        self.use_scr = use_scr
        # loss
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean').cuda()
#        self.clustering_loss = ClusteringLoss(temperature=0.1)  # 添加聚类损失实例

        # set backbone
        self.variant = variant
        self.trunk = trunk

        # proj
        self.cont_proj_head = cont_proj_head
        if wild_cont_dict_size > 0:
            if cont_proj_head > 0:
                self.cont_dict = {}
                self.cont_dict['size'] = wild_cont_dict_size
                self.cont_dict['dim'] = self.cont_proj_head

                self.register_buffer("wild_cont_dict", torch.randn(self.cont_dict['dim'], self.cont_dict['size']))
                self.wild_cont_dict = nn.functional.normalize(self.wild_cont_dict, p=2, dim=0)  # C X Q
                self.register_buffer("wild_cont_dict_ptr", torch.zeros(1, dtype=torch.long))
                self.cont_dict['wild'] = self.wild_cont_dict.cuda()
                self.cont_dict['wild_ptr'] = self.wild_cont_dict_ptr
            else:
                raise 'dimension of wild-content dictionary is zero'

        channel_1st = 4
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-50':
            resnet = resnet50(fs_layer=self.fs_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            os = 16
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unknown deepv3 variant: {}'.format(self.variant)

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)


        # self. msaa = MSAA(in_channels=256, out_channels=256)   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))




        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)  #这个辅助输出与self.final2的输出一起，可以提供更准确的分割结果
        )
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)









        # if self.cont_proj_head > 0: # 映射器（更改为卷积）
        #      self.proj = nn.Sequential(
        #         nn.Linear(256, 256, bias=True),
        #         nn.ReLU(inplace=False),
        #          nn.Linear(256, self.cont_proj_head, bias=True))
        #      initialize_weights(self.proj)

        if self.cont_proj_head > 0:  # 映射器（原论文）
            self.proj = nn.Sequential(
                nn.Linear(256, 256, bias=True),
                nn.ReLU(inplace=False),
                nn.Linear(256, self.cont_proj_head, bias=True))
            initialize_weights(self.proj)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-50':
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]
        else:
            raise ValueError("Not a valid network arch")




    def forward(self, x,x_t=None , gts=None,png=None, weights=None, apply_fs=True):

        # global low_level_st, low_level_t,aux_out_st,main_out_t,aux_out_t
        x_size = x.size()  # 800

        # encoder
        x = self.layer0[0](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_t = self.layer0[0](x_t)

        x = self.layer0[1](x)
        if self.training & apply_fs:
            x_st = self.layer0[1](x, x_t)  # feature stylization

            with torch.no_grad():
                x_t = self.layer0[1](x_t)
        x = self.layer0[2](x)
        x = self.layer0[3](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_t = self.layer0[2](x_t)
                x_t = self.layer0[3](x_t)
            x_st = self.layer0[2](x_st)
            x_st = self.layer0[3](x_st)


        if self.training & apply_fs:
            x_tuple = self.layer1([x, x_t, x_st])
            low_level = x_tuple[0]
            low_level_t = x_tuple[1]
            low_level_st = x_tuple[2]

        else:
            x_tuple = self.layer1([x])
            low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)

        x_tuple = self.layer3(x_tuple)

        aux_out = x_tuple[0]  #从layer3的输出中提取第一个元素作为辅助输出

        if self.training & apply_fs:   #如果模型处于训练模式并且特征风格化被启用，提取变换后和风格化后的辅助输出
            aux_out_t = x_tuple[1]
            aux_out_st = x_tuple[2]
        x_tuple = self.layer4(x_tuple)   #将可能已经包含变换后和风格化数据的x_tuple传递给layer4进行最终处理

        x = x_tuple[0]    #从layer4的输出中提取最终的主要输出特征
        if self.training & apply_fs:  #如果模型处于训练模式并且特征风格化被启用，提取变换后和风格化后的主要输出特征

            x_t = x_tuple[1]
            x_st = x_tuple[2]

        # decoder
        x = self.aspp(x)       #将特征图x通过ASPP（Atrous Spatial Pyramid Pooling）模块处理，以捕获多尺度的特征
        dec0_up = self.bot_aspp(x)       #将ASPP的输出x进一步通过bot_aspp层处理，通常是一个1x1的卷积层，用于降低维度
        dec0_fine = self.bot_fine(low_level)     #将低层特征low_level通过bot_fine层处理，以保留更多的空间细节


        dec0_up = Upsample(dec0_up, low_level.size()[2:]) #使用上采样（Upsample）将dec0_up的尺寸调整到与low_level相同，以便于特征融合
        dec0 = [dec0_fine, dec0_up]     #创建一个列表，包含经过细化和上采样的特征图
        dec0 = torch.cat(dec0, 1)      #沿着通道维度（维度1），将dec0列表中的两个特征图进行拼接
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)     #将final1的输出dec1通过final2层处理，通常是一个分类层，用于生成最终的语义分割图
        main_out = Upsample(dec2, x_size[2:])   #将dec2上采样到输入x的原始空间尺寸，得到最终的输出main_out

        if self.training:
            # compute original semantic segmentation loss
            loss_orig = self.criterion(main_out, gts)  #计算主输出main_out相对于真实标签gts的损失
            aux_out = self.dsn(aux_out)  #通过辅助分割网络（可能是一个额外的卷积层）处理aux_out，得到一个辅助输出
            aux_gts =png      #    aux_gts = png           #将png变量赋值给aux_gts，这里png可能是包含伪标签或辅助真实标签的张量
            aux_gts = aux_gts.unsqueeze(1).float()     #增加一个维度，并转换为浮点数类型，以满足损失函数的要求
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest') #使用最近邻插值将aux_gts上采样到与aux_out相同的空间尺寸
            aux_gts = aux_gts.squeeze(1).long() #移除aux_gts中增加的单维度，并转换为长整型
            loss_orig_aux = self.criterion_aux(aux_out, aux_gts, weights)  #计算辅助输出aux_out相对于辅助真实标签aux_gts的损失


            return_loss = [loss_orig, loss_orig_aux]
            # return_loss = [loss_orig]







            if apply_fs:
                x_st = self.aspp(x_st)
                dec0_up_st = self.bot_aspp(x_st)
                dec0_fine_st = self.bot_fine(low_level_st)
                dec0_up_st = Upsample(dec0_up_st, low_level_st.size()[2:])
                dec0_st = [dec0_fine_st, dec0_up_st]
                dec0_st = torch.cat(dec0_st, 1)
                dec1_st = self.final1(dec0_st)
                dec2_st = self.final2(dec1_st)
                main_out_st = Upsample(dec2_st, x_size[2:])

                with torch.no_grad():
                    x_t = self.aspp(x_t)
                    dec0_up_t = self.bot_aspp(x_t)
                    dec0_fine_t = self.bot_fine(low_level_t)
                    dec0_up_t = Upsample(dec0_up_t, low_level_t.size()[2:])
                    dec0_t = [dec0_fine_t, dec0_up_t]
                    dec0_t = torch.cat(dec0_t, 1)
                    dec1_t = self.final1(dec0_t)
                    dec2_t = self.final2(dec1_t)
                    main_out_t = Upsample(dec2_t, x_size[2:])





                if self.use_cel:
                    # projected features
                    assert (self.cont_proj_head > 0)
                    proj2 = self.proj(dec1.permute(0,2,3,1)).permute(0,3,1,2)  #在forward方法中，proj映射器被用于特征提取
                    proj2_st = self.proj(dec1_st.permute(0,2,3,1)).permute(0,3,1,2) #在forward方法中，proj映射器被用于特征提取
                    with torch.no_grad():
                        proj2_t = self.proj(dec1_t.permute(0,2,3,1)).permute(0,3,1,2)




                    # compute content extension learning loss
                    loss_cel = get_content_extension_loss(proj2, proj2_st, proj2_t, png, self.cont_dict)
                    # loss_cel = 0.5 * get_content_extension_loss(proj2, proj2_st, proj2_t, png,
                    #                                             self.cont_dict) + 0.5 * small_get_content_extension_loss(
                    #     proj2, proj2_st, proj2_t, png, self.cont_dict)
                    return_loss.append(loss_cel)




                    # # 确保特征图的空间尺寸匹配
                    # if proj2_st.size(2) != proj2_t.size(2) or proj2_st.size(3) != proj2_t.size(3):
                    #     # 使用双线性插值调整尺寸
                    #     proj2_st = torch.nn.functional.interpolate(proj2_st, size=(proj2_t.size(2), proj2_t.size(3)),
                    #                                                mode='bilinear', align_corners=False)
                    #
                    # # 展平特征张量到二维
                    # proj2_st_flat = proj2_st.reshape(proj2_st.size(0), -1)
                    # proj2_t_flat = proj2_t.reshape(proj2_t.size(0), -1)
                    #
                    # # 计算聚类损失
                    # cluster_loss = self.clustering_loss(proj2_st_flat, proj2_t_flat)
                    # return_loss.append(cluster_loss)


                # # 假设 dec2_t 是通过 self.final2(dec1_t) 得到的
                    # dec2_t = self.final2(dec1_t)
                    #
                    # # 确保 dec2_t 和 proj2_t 的空间尺寸匹配
                    # if dec2_t.size(2) != proj2_t.size(2) or dec2_t.size(3) != proj2_t.size(3):
                    #     # 使用双线性插值调整 dec2_t 的尺寸以匹配 proj2_t 的空间尺寸
                    #     dec2_t = torch.nn.functional.interpolate(dec2_t, size=proj2_t.size()[2:], mode='bilinear',
                    #                                              align_corners=False)
                    #
                    # # 确保 dec2_t 和 proj2_t 的通道数量匹配
                    # if dec2_t.size(1) != proj2_t.size(1):
                    #     # 创建一个卷积层来调整通道数量
                    #     adjust_channels = torch.nn.Conv2d(in_channels=dec2_t.size(1), out_channels=proj2_t.size(1),
                    #                                       kernel_size=1)
                    #     # 确保卷积层使用与特征图相同的设备
                    #     adjust_channels = adjust_channels.to(proj2_t.device)
                    #     # 调整 dec2_t 的通道数量
                    #     dec2_t = adjust_channels(dec2_t)
                    #
                    # # 展平特征图以计算损失
                    # batch_size, channels, height, width = dec2_t.shape
                    # num_features = channels * height * width
                    #
                    # # 使用 .reshape() 而不是 .view() 来展平张量
                    # dec2_t_flat = dec2_t.reshape(batch_size, num_features)
                    # proj2_t_flat = proj2_t.reshape(batch_size, num_features)
                    #
                    # # 计算 CC-Loss
                    # cc_loss = self.cc_consistency_loss(dec2_t_flat, proj2_t_flat)
                    #
                    # # 将损失添加到返回列表中
                    # return_loss.append(cc_loss)








                    # # 假设 proj2_t 是一个在 GPU 上的四维张量，形状为 [batch_size, channels, height, width]
                    # # 首先将张量移动到 CPU
                    # proj2_t_cpu = proj2_t.to('cpu')
                    #
                    # # 定义一个旋转变换，例如旋转 90 度
                    # rotation_transform = transforms.RandomRotation(degrees=90)
                    #
                    # # 应用数据增强
                    # # 需要将张量转换为 PIL 图像，应用变换后再转换回张量
                    # # 这里假设 proj2_t 的值在 [0, 1] 之间，且是 float 类型
                    # # 首先，我们需要将四维张量转换为三视张量
                    # proj2_t_pil = [transforms.ToPILImage()(img) for img in proj2_t_cpu.unbind(1)]
                    # proj2_t_rotated = [rotation_transform(img) for img in proj2_t_pil]
                    # proj2_t_rotated_tensor = torch.stack([transforms.ToTensor()(img) for img in proj2_t_rotated])
                    #
                    # # 将增强后的张量移回 GPU
                    # proj2_t_rotated_tensor = proj2_t_rotated_tensor.to('cuda')
                    #
                    # # 确保 proj2_t 和 proj2_t_rotated_tensor 都在 GPU 上
                    # proj2_t = proj2_t.to('cuda')
                    #
                    # # 展平特征图
                    # batch_size, channels, height, width = proj2_t.shape
                    # num_features = channels * height * width
                    #
                    # # 使用 .reshape() 而不是 .view() 来展平张量
                    # proj2_t_flat = proj2_t.reshape(batch_size, num_features)
                    # proj2_t_rotated_flat = proj2_t_rotated_tensor.reshape(batch_size, num_features)
                    #
                    # # 计算 CC-Loss
                    # cc_loss = self.cc_consistency_loss(proj2_t_flat, proj2_t_rotated_flat)
                    #
                    # # 将损失添加到返回列表中
                    # return_loss.append(cc_loss)



                # if self.training and self.use_cel:
                #     # 确保 self.cont_proj_head 大于 0，以便使用 proj 映射器
                #     assert (self.cont_proj_head > 0)
                #
                #     # 使用 proj 映射器提取特征
                #     proj2 = self.proj(dec1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                #     proj2_st = self.proj(dec1_st.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                #     with torch.no_grad():
                #         proj2_t = self.proj(dec1_t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                #
                #     # 计算内容扩展学习损失
                #     loss_cel = get_content_extension_loss(proj2, proj2_st, proj2_t, png, self.cont_dict)
                #
                #     return_loss.append(loss_cel)

                    # 在计算 CCConsistency 损失之前，将四维特征图 proj2 和 proj2_st 转换为二维张量
                    # 假设特征图的维度为 [batch_size, channels, height, width]
                    # batch_size, channels, height, width = proj2.shape
                    # # 假设 dec1 是原始的四维特征张量 [batch_size, channels, height, width]
                    # # 首先计算展平后的特征总数
                    # num_features = dec1.size(1) * dec1.size(2) * dec1.size(3)  # channels * height * width
                    #
                    # # 使用 view 方法直接展平四维张量到二维张量
                    # proj2_flat = dec1.view(batch_size, num_features)  # 展平 dec1
                    # proj2_st_flat = dec1_st.view(batch_size, num_features)  # 展平 dec1_st
                    #
                    # # 接下来，确保 CCConsistency 类的 forward 方法可以接受这些展平后的张量
                    # # 并正确处理它们
                    # cc_loss = self.cc_consistency_loss(proj2_flat, proj2_st_flat)
                    #
                    #
                    # return_loss.append( cc_loss )

                 ##########################################################################   # # 确保 proj2_st 和 proj2_t 的空间尺寸匹配
                    if proj2_st.size(2) != proj2_t.size(2) or proj2_st.size(3) != proj2_t.size(3):
                        # 使用双线性插值调整 proj2_st 的尺寸以匹配 proj2_t 的空间尺寸
                        proj2_st = torch.nn.functional.interpolate(proj2_st, size=proj2_t.size()[2:], mode='bilinear',
                                                                   align_corners=False)

                    # 确保 proj2_st 和 proj2_t 的通道数量匹配
                    if proj2_st.size(1) != proj2_t.size(1):
                        # 创建一个卷积层来调整通道数量
                        adjust_channels = torch.nn.Conv2d(in_channels=proj2_st.size(1), out_channels=proj2_t.size(1),
                                                          kernel_size=1)
                        # 确保卷积层使用与特征图相同的设备
                        adjust_channels = adjust_channels.to(proj2_t.device)
                        # 调整 proj2_st 的通道数量
                        proj2_st = adjust_channels(proj2_st)

                    # 展平特征图以计算损失
                    batch_size, channels, height, width = proj2_st.shape
                    num_features = channels * height * width

                    # 使用 .reshape() 而不是 .view() 来展平张量
                    proj2_st_flat = proj2_st.reshape(batch_size, num_features)
                    proj2_t_flat = proj2_t.reshape(batch_size, num_features)

                    # 计算 CC-Loss
                    cc_loss = self.cc_consistency_loss(proj2_t_flat, proj2_st_flat)

                    # 将损失添加到返回列表中
                    return_loss.append(cc_loss)

                # 假设 dec2_t 和 dec2_st 是我们想要比较的两个特征图
                # 假设 cc_consistency_loss 是已经定义好的计算 CC-Loss 的函数

                # with torch.no_grad():
                #     # 确保 dec2_st 和 dec2_t 的空间尺寸匹配
                #     if dec2_st.size(2) != dec2_t.size(2) or dec2_st.size(3) != dec2_t.size(3):
                #         # 使用双线性插值调整 dec2_st 的尺寸以匹配 dec2_t 的空间尺寸
                #         dec2_st = torch.nn.functional.interpolate(dec2_st, size=dec2_t.size()[2:], mode='bilinear',
                #                                                   align_corners=False)
                #
                #     # 确保 dec2_st 和 dec2_t 的通道数量匹配
                #     if dec2_st.size(1) != dec2_t.size(1):
                #         # 创建一个卷积层来调整通道数量
                #         adjust_channels = torch.nn.Conv2d(in_channels=dec2_st.size(1), out_channels=dec2_t.size(1),
                #                                           kernel_size=1)
                #         # 确保卷积层使用与特征图相同的设备
                #         adjust_channels = adjust_channels.to(dec2_t.device)
                #         # 调整 dec2_st 的通道数量
                #         dec2_st = adjust_channels(dec2_st)
                #
                #     # 展平特征图以计算损失
                #     batch_size, channels, height, width = dec2_st.shape
                #     num_features = channels * height * width
                #
                #     # 使用 .reshape() 而不是 .view() 来展平张量
                #     dec2_st_flat = dec2_st.reshape(batch_size, num_features)
                #     dec2_t_flat = dec2_t.reshape(batch_size, num_features)
                #
                #     # 计算 CC-Loss
                #     cc_loss = self.cc_consistency_loss(dec2_t_flat, dec2_st_flat)
                #
                #     # 将损失添加到返回列表中
                #     return_loss.append(cc_loss)




















































                if self.use_sel:
                    # compute style extension learning loss
                    loss_sel = self.criterion(main_out_st, gts)
                    return_loss.append(loss_sel)





                    aux_out_st = self.dsn(aux_out_st)
                    loss_sel_aux = self.criterion_aux(aux_out_st, aux_gts, weights)
                    return_loss.append(loss_sel_aux)

                if self.use_scr:
                    # compute semantic consistency regularization loss
                    loss_scr = torch.clamp((self.criterion_kl(nn.functional.log_softmax(main_out_st, dim=1), nn.functional.softmax(main_out, dim=1)))/(torch.prod(torch.tensor(main_out.shape[1:]))), min=0)
                    loss_scr_aux = torch.clamp((self.criterion_kl(nn.functional.log_softmax(aux_out_st, dim=1), nn.functional.softmax(aux_out, dim=1)))/(torch.prod(torch.tensor(aux_out.shape[1:]))), min=0)
                    return_loss.append(loss_scr)
                    return_loss.append(loss_scr_aux)

            return return_loss, main_out

        else:
            return main_out


def DeepR50V3PlusD(num_classes, criterion=Dice_loss, criterion_aux=CE_Loss, cont_proj_head=256,
                   cc_consistency_temperature=0.45, cc_consistency_thr=0.7):  #,cc_consistency_temperature=1.0, cc_consistency_thr=0.7   cc_consistency_temperature=0.45, cc_consistency_thr=0.7==zuihao   cc_consistency_temperature=0.45, cc_consistency_thr=0.75
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                      cont_proj_head=cont_proj_head, variant='D16', skip='m1', fs_layer=[1,0, 1, 0, 0], use_cel=True, use_sel=True, use_scr=True,cc_consistency_temperature=cc_consistency_temperature, cc_consistency_thr=cc_consistency_thr
)#,cc_consistency_temperature=cc_consistency_temperature, cc_consistency_thr=cc_consistency_thr
