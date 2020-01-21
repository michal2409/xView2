import torch.nn as nn
import torch

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def jaccard(preds, trues, weight=None, is_average=True, eps=1e-7):
    preds = preds[:, 1].float()
    trues = trues.float()
    
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return jaccard(input, target, self.weight, self.size_average)

def dice_loss(preds, trues, weight=None, is_average=True, eps=1e-7):
    preds = preds[:, 1].float()
    trues = trues.float()

    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return dice_loss(input, target, self.weight, self.size_average)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True, thr=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(weight=alpha)
        self.thr = thr

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma / self.thr**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
    
class ReducedFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True, thr=0.5):
        super(ReducedFocalLoss, self).__init__()
        self.gamma = gamma
        self.thr = thr
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        mask = (pt > self.thr).type(torch.FloatTensor).to(device)
        neq_mask = (1 - mask).type(torch.FloatTensor).to(device)
        coef = ((1-pt)**self.gamma / self.thr**self.gamma)*mask + neq_mask
        loss = coef * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

