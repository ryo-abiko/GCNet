import torch
import torch.nn as nn
from torchvision.utils import make_grid


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

        return x


def gridImage(imgs):

    img_grid = make_grid(imgs[0], nrow=1, normalize=False)

    for i in range(1,len(imgs)):
        img_grid = torch.cat((img_grid, make_grid(imgs[i], nrow=1, normalize=False)), -1)

    return img_grid