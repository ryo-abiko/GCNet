"""  Usage
net = generator()
GCnet = GCLoss()
criterion_edge_corr = torch.nn.L1Loss()
loss_edge_sum = torch.nn.L1Loss()

B, R = net(Input)
gradB, gradR = GCLoss(B, R)
loss_edge_corr = criterion_edge_corr(gradB * gradR, 0)
loss_edge_sum = criterion_edge_sum(gradB + gradR, imgs_grad_label)
loss_edge = (epoch - 1) * (loss_edge_sum + loss_edge_corr)
"""

import torch.nn as nn


class GCLoss(nn.Module):
    def __init__(self):
        super(GCLoss, self).__init__()

        sobel_x = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view((1,1,3,3)).repeat(1,3,1,1)
        sobel_y = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view((1,1,3,3)).repeat(1,3,1,1)

        self.G_x_B = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_x_B.weight = nn.Parameter(sobel_x)
        for param in self.G_x_B.parameters():
            param.requires_grad = False

        self.G_y_B = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_y_B.weight = nn.Parameter(sobel_y)
        for param in self.G_y_B.parameters():
            param.requires_grad = False


        self.G_x_R = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_x_R.weight = nn.Parameter(sobel_x)
        for param in self.G_x_R.parameters():
            param.requires_grad = False
        
        self.G_y_R = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_y_R.weight = nn.Parameter(sobel_y)
        for param in self.G_y_R.parameters():
            param.requires_grad = False

        self.af_B = nn.Tanhshrink()
        self.af_R = nn.Tanhshrink()
        

    def forward(self, B, R):

        gradout_B = self.af_B(self.G_y_B(B) + self.G_x_B(B))
        gradout_R = self.af_R(self.G_y_R(R) + self.G_x_R(R))
        return gradout_B, gradout_R