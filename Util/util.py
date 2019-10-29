import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Normfunc(nn.Module):
    def __init__(self):
        super(Normfunc, self).__init__()
        
    def forward(self, x):
        tempx = x.clone()
        for i in range(3):
            tempx[:,i,:,:] = (tempx[:,i,:,:] - mean[i]) / std[i]

        return tempx


class UnNormfunc(nn.Module):
    def __init__(self):
        super(UnNormfunc, self).__init__()
        
    def forward(self, x):
        tempx = x.clone()
        for i in range(3):
            tempx[:,i,:,:] = tempx[:,i,:,:] * std[i] + mean[i]

        return tempx


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