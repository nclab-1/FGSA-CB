"""
@author: Ying Jin
@contact: sherryying003@gmail.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.classifier import Classifier as ClassifierBase
from tllib.modules.entropy import entropy


def entropy(predictions):
    # 计算预测的熵
    return -torch.sum(predictions * torch.log(predictions + 1e-6), dim=-1)

class CCConsistency(nn.Module):
    def __init__(self, temperature: float, thr=0.7):
        super(CCConsistency, self).__init__()
        self.temperature = temperature
        self.thr = thr

    def forward(self, proj2_t_flat: torch.Tensor, proj2_st_flat : torch.Tensor) -> torch.Tensor:
        # 展平四维特征图到二维张量
        batch_size = proj2_t_flat.size(0)
        num_features =proj2_t_flat.view(batch_size, -1).size(1)
        logits =proj2_t_flat.view(batch_size, num_features)
        logits_strong = proj2_st_flat.view(batch_size, num_features)

        # 计算 softmax 概率分布
        prediction_thr = F.softmax(logits / self.temperature, dim=1)
        max_probs, max_idx = torch.max(prediction_thr, dim=-1)
        mask_binary = max_probs.ge(self.thr)  # 生成二进制掩码
        mask = mask_binary.float().detach()

        if mask.sum() == 0:
            return 0.0  # 如果没有预测高于阈值，返回0损失

        # 应用掩码
        logits = logits[mask_binary]
        logits_strong = logits_strong[mask_binary]

        # 计算概率分布
        predictions = F.softmax(logits / self.temperature, dim=1)
        predictions_strong = F.softmax(logits_strong / self.temperature, dim=1)

        # 计算熵权重
        entropy_weight = entropy(predictions).unsqueeze(1)  # 熵是沿着最后一个维度计算的
        entropy_weight_strong = entropy(predictions_strong).unsqueeze(1)

        # 调整熵权重
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight_strong = 1 + torch.exp(-entropy_weight_strong)

        # 计算加权的预测
        weighted_predictions = predictions * entropy_weight
        weighted_predictions_strong = predictions_strong * entropy_weight_strong

        # 计算类混淆矩阵
        class_confusion_matrix = torch.mm(weighted_predictions, predictions.transpose(1, 0))
        class_confusion_matrix /= torch.sum(class_confusion_matrix, dim=0, keepdim=True)

        class_confusion_matrix_strong = torch.mm(weighted_predictions_strong, predictions_strong.transpose(1, 0))
        class_confusion_matrix_strong /= torch.sum(class_confusion_matrix_strong, dim=0, keepdim=True)


#         # # 计算概率分布、熵权重和类混淆矩阵
#         # predictions = F.softmax(logits / self.temperature, dim=1)
#         # entropy_weight = entropy(predictions).detach()
#         # entropy_weight = 1 + torch.exp(-entropy_weight)
#         # entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
#         #
#         # class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions)
#         # class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
#         #
#         # predictions_stong = F.softmax(logits_strong / self.temperature, dim=1)
#         # entropy_weight_strong = entropy(predictions_stong).detach()
#         # entropy_weight_strong = 1 + torch.exp(-entropy_weight_strong)
#         # entropy_weight_strong = (batch_size * entropy_weight_strong / torch.sum(entropy_weight_strong)).unsqueeze(dim=1)
#         #
#         # class_confusion_matrix_strong = torch.mm((predictions_stong * entropy_weight_strong).transpose(1, 0), predictions_stong)
#         # class_confusion_matrix_strong = class_confusion_matrix_strong / torch.sum(class_confusion_matrix_strong, dim=1)

        # 计算一致性损失
        consistency_loss = ((class_confusion_matrix - class_confusion_matrix_strong) ** 2).sum() / num_features * mask.sum() / batch_size
        return consistency_loss