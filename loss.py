import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target):
        logs = torch.log(pred)
        for i in range(target.shape[0]):
            logs[i, target[i].item()] = 0
        return -torch.sum(logs, dim=1) / (pred.shape[1])