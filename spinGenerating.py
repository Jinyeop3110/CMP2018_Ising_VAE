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
import random

warnings.filterwarnings("ignore", ".*output shape of zoom.*")

class LatticeBasic(object):

    def __init__(self, N=16, T=1.0, J=1, interaction_h = None, interaction_v = None):
        self.N = N
        self.T = T
        self.lattice = None
        self.neighbor_filter = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        self.a=0
        self.b=J

        self.interaction_h = interaction_h
        self.interaction_v = interaction_v


        self.initialize()

    def initialize(self):
        """
        Initialize lattice points to -1 or 1 randomly
        """
        self.lattice = 2 * np.random.randint(2, size=(self.N, self.N)) - 1



    def step(self):
        """
        Every iteration, select N^2 random points to try a flip attempt.
        A flip attempt consists of checking the change in energy due to a flip.
        If it is negative or less than exp(-E/(k_b*T)), then perform the flip.
        """
        for istep in range(self.N ** 2):
            ix = np.random.randint(0, self.N)
            iy = np.random.randint(0, self.N)
            s = self.lattice[ix, iy]
            #neighbor_sum = self.lattice[(ix + 1) % self.N, iy] + \
            #               self.lattice[(ix - 1) % self.N, iy] + \
            #               self.lattice[ix, (iy + 1) % self.N] + \
            #               self.lattice[ix, (iy - 1) % self.N]
            neighbor_sum = self.lattice[(ix + 1) % self.N, iy] * self.interaction_h[ix, iy] + \
                           self.lattice[(ix - 1) % self.N, iy] * self.interaction_h[(ix - 1) % self.N, iy]+ \
                           self.lattice[ix, (iy + 1) % self.N] * self.interaction_v[ ix, iy]+ \
                           self.lattice[ix, (iy - 1) % self.N] * self.interaction_v[ ix, (iy - 1) % self.N]
            dE = self.a * 2 * s + self.b * 2 * s * neighbor_sum
            if dE < 0 or np.random.rand() < np.exp(-1.0 * dE / self.T):
                s *= -1
            self.lattice[ix, iy] = s

    def get_neighbor_sum_matrix(self):
        """
        While not as efficient as computing the energy once at the beginning
        and adding the dE every step(), this is quite *fast* and elegant.
        Use a 3x3 filter for adjacent neighbors and convolve this across
        the lattice. "wrap" boundary option will handle the periodic BCs.
        This returns a NxN matrix of the sum of neighbor spins for each point.
        """
        return convolve2d(self.lattice, self.neighbor_filter, mode="same", boundary="wrap")

    def get_energy(self):
        """
        We can write the hamiltonian using optimized operations now
        """
        return - self.a * self.lattice - self.b * (self.lattice * self.get_neighbor_sum_matrix()).sum()

    def get_avg_magnetization(self):
        return 1.0 * self.lattice.sum() / self.N ** 2

    def __repr__(self):
        return str(self.lattice)

class SpinDataset():
    # TODO : infer implementated
    def __init__(self, N=16, T_range = [1.0, 5.0],T_step=0.1, J=1, iteration = 100, torch_type="float", interaction_h= None, interaction_v= None):
        self.N=N
        self.T_range=T_range
        self.T_step=T_step
        self.J=J
        self.iteration=iteration
        self.torch_type = torch.float  if torch_type == "float" else torch.half
        self.interaction_h = interaction_h
        self.interaction_v = interaction_v
    
    def _np2tensor(self, np):
        tmp = torch.from_numpy(np)
        return tmp.to(dtype=self.torch_type)

    # TODO : infer implementated
    def _spin2D(self,T):
        input_=LatticeBasic(N=self.N, T = T, J = self.J, interaction_h=interaction_h, interaction_v=interaction_v)
        for _ in range(self.iteration):
            input_.step()

        return input_.lattice, T


    def save(self,f_path, Num):
        for T in self.T_range:
            for i in range(Num):
                data={}
                T_ = T + self.T_step * random.random()
                lattice, T_ = self._spin2D(T_)
                data["spin"] = lattice.astype(np.int8)
                data["T"] = T
                print(str(T) + " is done")
                scipy.io.savemat(f_path + "/" + "T=" + str(T) + '_' + str(i) + ".mat", data)



if __name__ == "__main__":
    # Test Data Loader
    f_path="/data1/0000Deployed_JY/IsingData"

    interaction_h = np.zeros([20, 20])
    interaction_v = 0.5 * np.ones([20, 20])
    for i in range(20):
        for j in range(20):
            interaction_h[i][j]=(1+pow(-1,i+j))/2
    data={}
    data['i_h']=interaction_h
    data['i_v']=interaction_v
    scipy.io.savemat("/home/jysong/PyCharmProjects_JY/CMP1122/outs/interactionData.mat", data)

    if(True)
    T_max = 5.0
    N_conf = 50
    T_step = T_max / N_conf
    T_range = np.arange(0, T_max, T_step)
    base = T_range[1]
    T_range = T_range + base
    N = 20
    f_path_addon = "/ladderinter_N20_a0_train"
    save_dir = f_path + f_path_addon
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    a = SpinDataset(N=N, T_range=T_range, T_step=T_step, J=1, iteration=150, interaction_h=interaction_h,
                    interaction_v=interaction_v)
    a.save(save_dir, 150)

    T_max = 5.0
    N_conf = 50
    T_step = T_max / N_conf
    T_range = np.arange(0, T_max, T_step)
    base = T_range[1]
    T_range = T_range + base
    N = 20
    f_path_addon = "/ladderinter_N20_a0_test"
    save_dir = f_path + f_path_addon
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    a = SpinDataset(N=N, T_range=T_range, T_step=T_step, J=1, iteration=150, interaction_h=interaction_h, interaction_v=interaction_v)
    a.save(save_dir, 20)

