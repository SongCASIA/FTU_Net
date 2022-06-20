import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DiceLoss(nn.Module):
    '''
    binary dice loss, 前景比例太小
    '''
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gt):
        smooth = 1e-5
        batch_size = pred.size(0)
        pred = F.sigmoid(pred)
        pred_flat = pred.view(batch_size, -1)
        gt_flat = gt.view(batch_size, -1)

        intersection = pred_flat * gt_flat
        dice_coefficient = 2 * (intersection.sum(1)) / (pred_flat.sum(1) + gt_flat.sum(1) + smooth)
        loss = 1 - dice_coefficient.sum() / batch_size

        return loss

class FocalLoss(nn.Module):
    '''
    binary focal loss
    '''
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, pred, gt):
        pred_flat = pred.view(-1)
        gt_flat = gt.view(-1)
        log_sigmoid = nn.LogSigmoid()
        log_pt = log_sigmoid(pred_flat)
        pt = torch.exp(log_pt)

        loss = -1 * self.alpha * ((1-pt)**self.gamma) * log_pt
        loss = loss.mean()

        return loss

class GHMC_Loss(nn.Module):
    def __init__(self, bins=10, momentum=0):
        super(GHMLoss, self).__init__()
        self.momentum = momentum
        self.bins = bins
        self.edges = torch.arange(bins+1).float() / bins
        self.bce_loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight)

    def forward(self, pred, gt):
        g = torch.abs(torch.sigmoid(pred).detach() - gt).detach()
        bin_idx = torch.floor(g * (self.bins - 0.0001)).long()
        bin_count = torch.zeros((self.bins))
        for i in range(self.bins):
            bin_count[i] = (bin_idx == i).sum().item()
        N = pred.size(0) * pred.size(1)
        nonempty_bins = (bin_count > 0).sum().item()
        GD = bin_count * nonempty_bins
        GD = torch.clamp(GD, min=0.0001)
        weight = 1/ GD
        loss = self.bce_loss(pred, target, weight[bin_idx])

        return loss

class structure_loss(nn.Module):
    """
    structure loss in TransFuse
    """
    def __init__(self):
        super(structure_loss, self).__init__()

    def forward(self, pred, gt):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        wbce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * gt) * weit).sum(dim=(2, 3))
        union = ((pred + gt) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        loss = (wbce + wiou).mean()

        return loss

class Multi_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Multi_DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes