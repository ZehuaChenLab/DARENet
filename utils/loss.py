import torch
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, smooth = 0., sigmoid: bool = True):
        super(dice_bce_loss, self).__init__()
        self.sigmoid = sigmoid
        self.smooth = smooth
        if sigmoid:
            self.bce = torch.nn.BCEWithLogitsLoss()
        else:
            self.bce = torch.nn.BCELoss()
        self.bce_loss = 0.0
        self.dice_loss = 0.0

    def soft_dice_loss(self, y_true , y_pred):
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(y_true * y_pred)
        coeff = (2. * intersection + self.smooth) / (torch.sum(y_true) + torch.sum(y_pred) + self.smooth)
        return 1. - coeff.mean()

    def update(self):
        self.bce_loss = 0.0
        self.dice_loss = 0.0

    def get_loss(self):
        return {'bce_loss ': self.bce_loss,
                'dice_loss': self.dice_loss}

    def forward(self, y_true, y_pred):
        bce = self.bce(y_pred, y_true)
        self.bce_loss += bce
        dice = self.soft_dice_loss(y_true, y_pred)
        self.dice_loss += dice
        loss = bce + dice
        return loss
