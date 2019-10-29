import glob
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class testImageDataset(Dataset):
    def __init__(self, root):
        
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/input/*.*"))

    def __getitem__(self, index):
        filePath = self.files[index % len(self.files)]
        R = np.array(Image.open(filePath),'f') / 255.

        return {"R": self.tensor_setup(R[:,:,:3]), "Name": os.path.basename(filePath).split(".")[0]}

    def __len__(self):
        return len(self.files)



class gtTestImageDataset(Dataset):
    def __init__(self, root):
        
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
  
        self.files_base = sorted(glob.glob(root + "/gt/*.*"))
        self.files_input = sorted(glob.glob(root + "/input/*.*"))

    def __getitem__(self, index):
        filePath = self.files_base[index % len(self.files_base)]
        B = np.array(Image.open(filePath),'f') / 255.
        R = np.array(Image.open(self.files_input[index % len(self.files_base)]),'f') / 255.

        return {"R": self.tensor_setup(R[:,:,:3]), "B": B[:,:,:3], "Name": os.path.basename(filePath).split(".")[0]}

    def __len__(self):
        return len(self.files_base)