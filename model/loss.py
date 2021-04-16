import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class MonaiLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.focal = FocalLoss(gamma=2.0)
        self.dice_bg = DiceLoss(include_background=True, softmax=True, to_onehot_y=True, batch=True)
        self.dice_nbg = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, batch=True)

    def forward(self, y_pred, y_true):
        y_true = y_true.unsqueeze(1).float()
        if self.loss == "dice":
            if y_pred.shape[1] == 2:
                return self.dice_nbg(y_pred, y_true)
            return self.dice_bg(y_pred, y_true)
        return self.focal(y_pred, y_true)


class Ohem(nn.Module):
    # Based on https://arxiv.org/pdf/1812.05802.pdf
    def __init__(self, fraction=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.fraction = fraction

    def forward(self, y_pred, y_true):
        batch_size = y_true.size(0)
        losses = self.loss(y_pred, y_true).view(batch_size, -1)

        positive_mask = (y_true > 0).view(batch_size, -1)
        Cp = torch.sum(positive_mask, dim=1)
        Cn = torch.sum(~positive_mask, dim=1)
        Chn = torch.max((Cn / 4).clamp_min(5), 2 * Cp)

        loss, num_samples = 0, 0
        for i in range(batch_size):
            positive_losses = losses[i, positive_mask[i]]
            negative_losses = losses[i, ~positive_mask[i]]
            num_negatives = int(Chn[i])
            hard_negative_losses, _ = negative_losses.sort(descending=True)[:num_negatives]
            loss = positive_losses.sum() + hard_negative_losses.sum() + loss
            num_samples += positive_losses.size(0)
            num_samples += hard_negative_losses.size(0)
        loss /= float(num_samples)

        return loss


class CORAL(nn.Module):
    # Adapted to image segmentation based on https://github.com/Raschka-research-group/coral-cnn
    def __init__(self):
        super(CORAL, self).__init__()
        self.levels = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)

    def forward(self, y_pred, y_true):
        device = y_pred.device
        levels = self.levels[y_true].to(device)
        logpt = F.logsigmoid(y_pred)
        loss = torch.sum(logpt * levels + (logpt - y_pred) * (1 - levels), dim=1)
        return -torch.mean(loss)


losses = {
    "dice": MonaiLoss("dice"),
    "focal": MonaiLoss("focal"),
    "ce": nn.CrossEntropyLoss(),
    "ohem": Ohem(),
    "mse": nn.MSELoss(),
    "coral": CORAL(),
}


class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_str = args.loss_str
        self.post = args.type == "post"
        self.losses = nn.ModuleList([losses[loss_fn] for loss_fn in self.loss_str.split("+")])

    def forward(self, y_pred, y_true):
        if self.post:
            device = y_pred.device
            mask = y_true > 0
            y_pred = torch.stack([y_pred[:, i][mask] for i in range(y_pred.shape[1])], 1).to(device)
            y_true = y_true[mask] - 1

        if self.loss_str == "mse":
            y_pred = F.relu(y_pred[:, 0], inplace=True)
            y_true = y_true.float()
        else:
            y_true = y_true.long()

        loss = 0
        for loss_fn in self.losses:
            loss += loss_fn(y_pred, y_true)
        return loss
