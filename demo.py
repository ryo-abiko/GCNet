
import os
import argparse
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from Util.util import Interpolate, gridImage
from GCNet_model import GCNet
from Util.dataset import gtTestImageDataset, testImageDataset, mean, std, padsize

def convert_to_numpy(input,H,W):
    image = input[:,:,padsize:H-padsize,padsize:W-padsize].clone()
    input_numpy = image[:,:,:H,:W].clone().cpu().numpy().reshape(3,H-padsize*2,W-padsize*2).transpose(1,2,0)
    for i in range(3):
        input_numpy[:,:,i] = input_numpy[:,:,i] * std[i] + mean[i]

    return  input_numpy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="usrImg", help="name of folder")
opt = parser.parse_args()

# make output directory
os.makedirs("images/" + opt.dataset_name + "/output", exist_ok=True)

# GPU or CPU
if torch.cuda.is_available():
  device = 'cuda'
  torch.backends.cudnn.benchmark=True
else:
  device = 'cpu'

# Initialize generator
Generator = GCNet().to(device)
Generator.eval()
Generator.load_state_dict(torch.load("GCNet_weight.pth", map_location=device))

# read image
gtAvailable = False
if os.path.exists("images/" + opt.dataset_name + "/gt"):
    if len(os.listdir("images/" + opt.dataset_name + "/input")) == len(os.listdir("images/" + opt.dataset_name + "/gt")):
        gtAvailable = True

if gtAvailable:
    image_dataset = gtTestImageDataset("images/" + opt.dataset_name)
else:
    image_dataset = testImageDataset("images/" + opt.dataset_name)

# run
all_psnr = 0.0
all_ssim = 0.0
print("[Dataset name: %s] --> %d images" % (opt.dataset_name, len(image_dataset)))
for image_num in tqdm(range(len(image_dataset))):

    data = image_dataset[image_num]
    R = data["R"].to(device)

    # Pad R for UNet
    _,first_h,first_w = R.size()
    R = torch.nn.functional.pad(R,(0,(R.size(2)//16)*16+16-R.size(2),0,(R.size(1)//16)*16+16-R.size(1)),"constant")
    R = R.view(1,3,R.size(1),R.size(2))

    # Process image
    with torch.no_grad():
        output  = Generator(R) 
        """
        output  = ( Generator(R)
                    + torch.rot90(Generator(torch.rot90(R,1,[2,3])),3,[2,3])
                    + torch.rot90(Generator(torch.rot90(R,3,[2,3])),1,[2,3])
                    + torch.rot90(Generator(torch.rot90(R,2,[2,3])),2,[2,3])
        ) / 4
        """

    # Process the output image
    output_np = np.clip(convert_to_numpy(output,first_h,first_w) + 0.015,0,1)
    R_np = convert_to_numpy(R,first_h,first_w)
    final_output = np.fmin(output_np, R_np)

    # Save image
    Image.fromarray(np.uint8(final_output * 255)).save("images/" + opt.dataset_name + "/output/" + data["Name"] + ".png")

    # Calculate PSNR/SSIM if available
    if gtAvailable:
        B = data["B"].astype(np.float32)
        thisPSNR = compare_psnr(B, final_output.astype(np.float32))
        thisSSIM = compare_ssim(B, final_output.astype(np.float32), multichannel=True)
        all_psnr += thisPSNR
        all_ssim += thisSSIM
        print("[%s] PSNR:%4.2f SSIM:%4.3f" % (data["Name"], thisPSNR, thisSSIM))

if gtAvailable:
    all_psnr = all_psnr/len(image_dataset)
    all_ssim = all_ssim/len(image_dataset)
    print("[[[[[[[[%s]]]]]]]]" % (opt.dataset_name))
    print("PSNR: %4.2f / SSIM: %4.3f" % (all_psnr, all_ssim))
else:
    print("Complete.")