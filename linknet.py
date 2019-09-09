import torch
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=768,
                 n_filters=256,
                 kernel_size=4,
                 nonlinearity=nn.ReLU):
        super().__init__()

        if kernel_size == 3:
            conv_stride = 1
        elif kernel_size == 1:
            conv_stride = 1
        elif kernel_size == 4:
            conv_stride = 2

        self.decoder = nn.Sequential([
            # B, C, L -> B, C/4, L
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      3,
                      padding=1),
            nn.BatchNorm1d(in_channels // 4),
            nonlinearity(inplace=True),
            # B, C/4, L -> B, C/4, L
            nn.ConvTranspose1d(in_channels // 4,
                               in_channels // 4,
                               kernel_size,
                               stride=conv_stride,
                               padding=1),
            nn.BatchNorm1d(in_channels // 4),
            nonlinearity(inplace=True),
            # B, C/4, L -> B, C, L
            nn.Conv1d(in_channels // 4,
                      n_filters,
                      3,
                      padding=1),
            nn.BatchNorm1d(n_filters),
            nonlinearity(inplace=True)
        ])

    def forward(self, x):
        return self.decoder(x)


class DenoiseLoss(nn.Module):
    def __init__(self):
        super(DenoiseLoss, self).__init__()

    def forward(self, output, target):
        mse_loss = nn.MSELoss(reduction=None)(output, target)
        bce_loss = nn.BCEWithLogitsLoss(reduction=None)(output, target)
        return mse_loss + bce_loss


class MaskSimilarity(nn.Module):
    def __init__(self,
                 thresholds=[0.05, 0.1, 0.15]):
        super().__init__()
        self.thresholds = thresholds

    def forward(self, outputs, targets):
        assert outputs.size() == targets.size()

        dice_output = nn.functional.sigmoid(outputs)
        abs_diff = torch.abs(dice_output - targets.float())
        total = (abs_diff < 1.0).sum()

        metrics = []
        for threshold in self.thresholds:
            similar = (abs_diff < threshold).sum()
            metrics.append(similar / total)

        return sum(metrics) / len(metrics)
