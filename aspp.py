import torch
from torch import nn
from torchvision.models.segmentation.deeplabv3 import ASPPConv, ASPPPooling

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, aspp_pooling=True, out_channels=2048):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        for r in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, r))
        if aspp_pooling:
            modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d((1 + int(aspp_pooling) + len(atrous_rates)) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)