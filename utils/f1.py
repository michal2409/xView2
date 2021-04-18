import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric


def convert_to_labels(loss_str, logits):
    if loss_str == "mse":
        preds = torch.round(F.relu(logits[:, 0], inplace=True)) + 1
        preds[preds > 4] = 4
    elif loss_str == "coral":
        preds = torch.sum(torch.sigmoid(logits) > 0.5, dim=1) + 1
    else:
        preds = torch.argmax(logits, dim=1) + 1
    return preds


class F1(Metric):
    def __init__(self, args):
        super().__init__(dist_sync_on_step=False)
        self.loss_str = args.loss_str
        self.n_class = 2 if args.type == "pre" else 5
        self.softmax = nn.Softmax(dim=1)
        self.add_state("tp", default=torch.zeros((self.n_class - 1,)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros((self.n_class - 1,)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros((self.n_class - 1,)), dist_reduce_fx="sum")

    def update(self, preds, targets):
        probs = self.softmax(preds) if self.loss_str not in ["mse", "coral"] else preds
        if self.n_class == 5:
            preds = convert_to_labels(self.loss_str, probs)
            mask = targets > 0
            targets = targets[mask]
            preds = preds[mask]
        else:
            preds = torch.argmax(probs, dim=1)

        for i in range(self.n_class - 1):
            true_pos, false_neg, false_pos = self.get_stats(preds, targets, i + 1)
            self.tp[i] += true_pos
            self.fn[i] += false_neg
            self.fp[i] += false_pos

    def compute(self):
        f1_score = 200 * self.tp / (2 * self.tp + self.fp + self.fn)
        if self.n_class == 5:
            f1 = 4 / sum((f1_ + 1e-6) ** -1 for f1_ in f1_score)
            return f1.cpu(), f1_score.cpu()
        return f1_score.cpu(), None

    @staticmethod
    def get_stats(pred, targ, class_idx):
        true_pos = torch.logical_and(pred == class_idx, targ == class_idx).sum()
        false_neg = torch.logical_and(pred != class_idx, targ == class_idx).sum()
        false_pos = torch.logical_and(pred == class_idx, targ != class_idx).sum()
        return true_pos, false_neg, false_pos
