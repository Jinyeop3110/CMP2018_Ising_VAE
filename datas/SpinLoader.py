import torch
import torch.utils.data as data
import random
import numpy as np

import os
import scipy.misc
from glob import glob
from scipy import io
import copyreg

# ignore skimage zoom warning
import warnings

import time
import pickle

import numpy as np
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", ".*output shape of zoom.*")


class SpinDataset(data.Dataset):

    def __init__(self, img_root, torch_type="float"):

        self.torch_type = torch.float  if torch_type == "float" else torch.half
        img_paths = glob(img_root + '/*.mat')
        if len(img_paths) == 0:
            raise ValueError("Check data path : %s" % (img_root))

        self.origin_image_len = len(img_paths)
        self.img_paths = img_paths
        self.torch_type = torch.float if torch_type == "float" else torch.half

    def _np2tensor(self, np):
        tmp = torch.from_numpy(np)
        return tmp.to(dtype=self.torch_type)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        data_pack = io.loadmat(img_path)
        # 2D ( 1 x H x W )
        input_np = data_pack['spin']
        input_np = input_np.astype(float)
        T_np = data_pack['T']

        input_ = self._np2tensor(input_np[:, :].copy())
        T = self._np2tensor(T_np)
        return input_[None], T

    def __len__(self):
        return len(self.img_paths)

    
def make_weights_for_balanced_classes(seg_dataset):
    count = [0, 0] # No mask, mask
    for img, mask in seg_dataset:
        count[int((mask > 0).any())] += 1

    N = float(sum(count))
    weight_per_class = [N / c for c in count]

    weight = [0] * len(seg_dataset)
    for i, (img, mask) in enumerate(seg_dataset):
        weight[i] = weight_per_class[int((mask > 0).any())]

    return weight, count

def SpinLoaderBasic(img_root, batch_size=8, torch_type="float", cpus=1, infer=False,
                  shuffle=True, drop_last=True):

    dataset = SpinDataset(img_root, torch_type=torch_type)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=cpus, drop_last=drop_last)

if __name__ == "__main__":
    # Test Data Loader
    f_path="/home/Jinyeop/PyCharmProjects_JY/180818_3DcellSegmentation_JY_ver1/data/"
    test_loader= nucleusloader(f_path , 10, shuffle=False, drop_last=False)

    for i, (input_, target_, fname) in enumerate(test_loader):
        print(fname, input_.shape, target_.dtype)
