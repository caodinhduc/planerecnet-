import torch
import torch.nn as nn

class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels*4, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.conv_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.conv_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)
    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        features_6 = self.conv_6(x)
        features_12 = self.conv_12(x)
        features_18 = self.conv_18(x)
        out = torch.cat([features, features_6, features_12, features_18], 1) 
        return torch.mul(out, attention_mask)