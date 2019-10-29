
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import shutil
from skimage.measure import compare_ssim, compare_psnr
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Util.util import *

def convert_to_numpy(input,H,W):
    
    return  input[:,:,:H,:W].clone().cpu().numpy().reshape(3,H,W).transpose(1,2,0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="synthetic" , help="name of the dataset")
parser.add_argument("--test_image", type=int, default=-1, help="test image number")
parser.add_argument("--epoch", type=int, default=-1, help="model epochs")
parser.add_argument("--train_name", type=str, default="test31", help="training name")
opt = parser.parse_args()


shutil.copyfile("models/models-" + opt.train_name  + ".py", "models.py")
from models import GeneratorUNet
from datasets1 import gtTestImageDataset
from trained_generator.GCNet_model import GCNet

if torch.cuda.is_available():
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'


# Initialize generator
preGenerator = GCNet().to(device)
generator = GeneratorUNet().to(device)

# Set generator to eval mode
preGenerator.eval()
generator.eval()


# load preGenerator
preGenerator.load_state_dict(torch.load( "trained_generator/GCNet_weight.pth"))

# Set up function
Norm = Normfunc()
UnNorm = UnNormfunc()


# load trained network model
if opt.epoch > -1:
    epc_list = [opt.epoch]
else:
    for i in range(210):
        if os.path.exists("saved_models/"+opt.train_name +"/generator_"+str(i)+".pth") == False:
            epc_list = range(i - 1)
            break


# read image
image_dataset = gtTestImageDataset("../../image/test/testdata_reflection_" + opt.dataset_name)
if opt.test_image > -1:
    img_list = [opt.test_image]
else:
    img_list = range(len(image_dataset))

# Set parameters 
Rmap_threshold = torch.tensor(0.6, device=device, requires_grad=False)

# data for finding model
bestPSNR = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]] # epoch, PSNR, SSIM

for epoch_num in epc_list:
    print("[[ Epoch: %d / %d]]" % (epoch_num,len(epc_list)))
    generator.load_state_dict(torch.load( "saved_models/"+opt.train_name+"/generator_"+str(epoch_num)+".pth"))

    all_psnr = 0.0
    all_ssim = 0.0
    for image_num in img_list:

        if opt.epoch > 0:
            print("[[ Image: %d / %d]]" % (image_num,len(image_dataset)))

        imgs = image_dataset[image_num]
        RF = imgs["RF"].to(device)
        B = imgs["B"]
        _,first_h,first_w = RF.size()
        RF = torch.nn.functional.pad(RF,(0,(RF.size(2)//16)*16+16-RF.size(2),0,(RF.size(1)//16)*16+16-RF.size(1)),"constant")
            
        opt.hr_height = RF.size(1)
        opt.hr_width = RF.size(2)
        RF = RF.view(1,3,opt.hr_height,opt.hr_width)


        with torch.no_grad():
            # Generate Reliability map
            GCNetOutput = UnNorm(preGenerator(Norm(RF)))
            Rmap = torch.abs(RF - GCNetOutput)
            maxNum = torch.max( Rmap )
            #Rmap = torch.pow(1 - (Rmap / maxNum), 2).detach()
            Rmap = torch.max(torch.pow(1 - (Rmap / maxNum), 2), Rmap_threshold).detach()
            #Rmap = (1 - (Rmap / maxNum)).detach()

            # Generate second estimated image / output is normalized in range 0-1
            input_tensor = torch.cat([ Norm(RF), Rmap, Norm((RF * Rmap)) ], 1)
            if opt.epoch > -1:
                output  = ( generator(input_tensor)[0]
                            + torch.rot90(generator(torch.rot90(input_tensor,1,[2,3]))[0],3,[2,3])
                            + torch.rot90(generator(torch.rot90(input_tensor,3,[2,3]))[0],1,[2,3])
                            + torch.rot90(generator(torch.rot90(input_tensor,2,[2,3]))[0],2,[2,3])
                ) / 4

            else:
                output  = generator(input)[0]


            
        # process the output image
        output = np.clip(convert_to_numpy(output,first_h,first_w),0,1)
        RF = convert_to_numpy(RF,first_h,first_w)
        #B = convert_to_numpy(B,first_h,first_w).astype(np.float32)


        final_output = output.astype(np.float32)
        final_psnr = compare_psnr(B, final_output)
        final_ssim = compare_ssim(B, final_output, multichannel=True)

        if opt.epoch > 0:
            print("            final psnr               -->         %4.2f" % (final_psnr))
        all_psnr += final_psnr
        all_ssim += final_ssim

    ave_PSNR = all_psnr/len(img_list)
    ave_SSIM = all_ssim/len(img_list)
    print("All dataset average PSNR: %4.2f" % (ave_PSNR))
    print("All dataset average SSIM: %4.3f" % (ave_SSIM))

    # ranking
    for i in range(5):
        if bestPSNR[i][1] <  ave_PSNR:
            bestPSNR.insert(i,[epoch_num, ave_PSNR, ave_SSIM])
            bestPSNR = bestPSNR[:5]
            break


print("            [[[[[[[[[[[  final result  %s  %s]]]]]]]]]]]" % (opt.train_name, opt.dataset_name))
for i in range(5):
    if bestPSNR[i][1] > 0:
        print( "[%d] epoch:%d  PSNR: %4.2f SSIM: %4.3f" % (i, bestPSNR[i][0], bestPSNR[i][1], bestPSNR[i][2]) )

if opt.test_image > 0:
    plt.figure(1).clear()
    plt.imshow(np.concatenate( [RF,output,B],1 ))
    plt.title(opt.train_name + " / " + str(opt.epoch) + "epochs")
    plt.show()