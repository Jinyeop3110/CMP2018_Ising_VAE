from multiprocessing import Process
import os
import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import utils
import scipy
torch.backends.cudnn.benchmark = True

# example for mnist
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/Jinyeop/PyCharmProjects_JY/180801_2DcellSegmentation_JY')

#from slack_server import SlackBot

from Logger import Logger

from models.VAE import VAEFC,VAEFC_2
from models.VAE_CONV import VAECONV
from models.GMVAE_DEEP import GMVAEFC_DEEP
from models.GMVAE import GMVAEFC
from datas.SpinLoader import SpinLoaderBasic
from trainers.ClusterTrainer_1125 import ClusterTrainer_1125

import copyreg
from scipy import io

from loss import Loss_Function_VAE

"""parsing and configuration"""
def arg_parse():
    desc = "SJY spin A.I"
    # projects description
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='spin',
                        choices=["spin", "GMVAE","GMVAE_DEEP"], required=True)
    # Unet params
    parser.add_argument('--feature_scale', type=int, default=4)

    parser.add_argument('--in_channel', type=int, default=1)

    # FusionNet Parameters
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')

    # TODO : Weighted BCE
    parser.add_argument('--loss', type=str, default='l1',
                        choices=["l1", "l2"])

    #parser.add_argument('--data', type=str, default='data',
    #                    choices=['All', 'Balance', 'data', "Only_Label"],
    #                    help='The dataset | All | Balance | Only_Label |')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--test', type=int, default=0, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    # Adam Parameter
    parser.add_argument('--lrG', type=float, default=0.00001)
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

    return parser.parse_args()


def reconstruct_torch_dtype(torch_dtype: str):
    # a dtype string is "torch.some_dtype"
    dtype = torch_dtype.split('.')[1]
    return getattr(torch, dtype)


def pickle_torch_dtype(torch_dtype: torch.dtype):
    return reconstruct_torch_dtype, (str(torch_dtype),)

if __name__ == "__main__":
    arg = arg_parse()
    arg.save_dir = "1215_laddar_VAE_z2_k3"

    arg.save_dir = "%s/outs/%s"%(os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
            os.mkdir(arg.save_dir)
    
    logger = Logger(arg.save_dir)

    copyreg.pickle(torch.dtype, pickle_torch_dtype)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")


    N=20
    hidden_dim=128
    z_dim=10
    cluster_k=3
    f_path_train="/data1/0000Deployed_JY/IsingData/Ising_N20_a0"
    f_path_test="/data1/0000Deployed_JY/IsingData/Ising_N20_a0_test"

    f_path_train = "/data1/0000Deployed_JY/IsingData/AFIsing_N20_a0"
    f_path_test = "/data1/0000Deployed_JY/IsingData/AFIsing_N20_a0_test"

    f_path_train = "/data1/0000Deployed_JY/IsingData/ladderinter_N20_a0_train"
    f_path_test = "/data1/0000Deployed_JY/IsingData/ladderinter_N20_a0_test"


    train_loader = SpinLoaderBasic(img_root=f_path_train, batch_size=arg.batch_size, shuffle=True)
    test_loader = SpinLoaderBasic(img_root=f_path_test, batch_size=1,shuffle=False)

    if arg.model == "GMVAE":
        net=GMVAEFC(N=N, hidden_dim=hidden_dim, z_dim=z_dim, k=cluster_k)
    elif arg.model == "GMVAE_DEEP":
        net=GMVAEFC_DEEP(N=N, hidden_dim=hidden_dim, z_dim=z_dim, k=cluster_k)

    else:
        raise NotImplementedError("Not Implemented Model")

    net = nn.DataParallel(net).to(torch_device)
    recon_loss=Loss_Function_VAE(N=N)

    if(True):
        model = ClusterTrainer_1125(arg, net, torch_device, recon_loss=recon_loss, val_loss=recon_loss, logger=logger)

        filename="epth.tar"
        model.load(filename=filename)
        model.best_metric=10000
        if arg.test==0:
            model.TrainWithHistorySave(train_loader, test_loader, save_dir= 'Cdata')
        if arg.test==1:
            model.test(test_loader, savedir=filename +'test')
        if arg.test==2:
            data = {}
            for name, param in model.G.named_parameters():
                if param.requires_grad:
                    print(name)
                    print((param.data.type(torch.FloatTensor)).numpy())
                    data[name.replace(".","")]=(param.data.type(torch.FloatTensor)).numpy()
            scipy.io.savemat(arg.save_dir + "/" + "model.mat", data)
        if arg.test==3:
            data = {}
            data['Img']=[]
            data['z1']=[]
            data['z2']=[]
            n = 8
            quantile_min = 0.01
            quantile_max = 0.99
            latent_dim = 2

            img_rows = 40
            img_cols = 40

            z1_u = np.linspace(2, -2, n)
            z2_u = np.linspace(2, -2, n)

            for z1 in z1_u:
                for z2 in z2_u:
                    z_=np.reshape([z1,z2],[1,z_dim])
                    data['Img'].append(np.reshape(np.squeeze((model.latent_sample(z_, N))),[N,N]))
                    data['z1'].append(z1)
                    data['z2'].append(z2)

            scipy.io.savemat(arg.save_dir + "/" + "model.mat", data)



