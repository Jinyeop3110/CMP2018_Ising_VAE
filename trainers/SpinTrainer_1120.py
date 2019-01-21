import os
from multiprocessing import Pool, Queue, Process

import scipy
import utils

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer
import torch.nn.functional as F
from scipy import io

# from sklearn.metrics import f1_score, confusion_matrix, recall_score, jaccard_similarity_score, roc_curve, precision_recall_curve

class SpinTrainer_1120(BaseTrainer):
    def __init__(self, arg, G, torch_device, recon_loss, val_loss, logger):
        super(SpinTrainer_1120, self).__init__(arg, torch_device, logger)
        self.recon_loss = recon_loss
        self.val_loss=val_loss
        
        self.G = G
        self.optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=arg.beta)
            
        self.best_metric = 10.0

        self.sigmoid = nn.Sigmoid().to(self.torch_device)

        self.load()
        self.prev_epoch_loss = 0


    def save(self, epoch, filename="models"):

        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        torch.save({"model_type" : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network" : self.G.state_dict(),
                    "optimizer" : self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_path + "/%s.pth.tar"%(filename))
        print("Model saved %d epoch"%(epoch))


    def load(self, filename="models.pth.tar"):
        if os.path.exists(self.save_path + "/" + filename) is True:
            print("Load %s File"%(self.save_path))            
            ckpoint = torch.load(self.save_path + "/" + filename)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))

            self.G.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d"%(ckpoint["model_type"], self.start_epoch))
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader=None, epoch=10000):
        print("\nStart Train")
        self.epoch=epoch

        criterion_inter=nn.CrossEntropyLoss()

        for epoch in range(self.start_epoch, self.epoch):
            for i, (input_, T) in enumerate(train_loader):
                self.G.train()
                input_= input_.to(self.torch_device)

                recon_batch, mu, logvar= self.G(input_)
                BCE,KLD = self.recon_loss(recon_batch, input_, mu, logvar)
                recon_loss = BCE+KLD
                self.optim.zero_grad()
                grads = {}
                def save_grad(name):
                    def hook(grad):
                        grads[name] = grad

                    return hook

                recon_loss.register_hook(save_grad('y'))

                recon_loss.backward()
                print(grads['y'])

                self.optim.step()
            
                if (i % 1) == 0:
                    self.logger.will_write("[Train] epoch:%d loss:%f" % (epoch, recon_loss))
            if val_loader is not None:            
                self.valid(epoch, val_loader)
            else:
                self.save(epoch)
        print("End Train\n")

    def _test_foward(self, input_, target_):
        input_  = input_.to(self.torch_device)
        output_= self.G(input_)
        target_ = target_
        input_  = input_

        return input_, output_, target_

    # TODO : Metric 정하기 
    def valid(self, epoch, val_loader):
        self.G.eval()
        with torch.no_grad():
            losssum=0
            count=0;

            for i, (input_, T) in enumerate(val_loader):
                if (i >= 1000):
                    break
                input_= input_.to(self.torch_device)

                recon_batch, mu, logvar= self.G(input_)
                BCE,KLD = self.val_loss(recon_batch, input_, mu, logvar)
                losssum=losssum+BCE+KLD
                count=count+1

            if losssum/count < self.best_metric:
                self.best_metric = losssum/count
                self.save(epoch,"epoch[%04d]_losssum[%f]"%(epoch, losssum/count))

            self.logger.write("[Val] epoch:%d losssum:%f "%(epoch, losssum/count))
                    
    # TODO: Metric, save file 정하기
    def test(self, test_loader, savedir=None):
        print("\nStart Test")
        self.G.eval()

        if savedir==None:
            savedir='/result/test'
        else:
            savedir='/result/'+savedir

        if os.path.exists(self.save_path+'/result') is False:
            os.mkdir(self.save_path + '/result')
        if os.path.exists(self.save_path+savedir) is False:
            os.mkdir(self.save_path+savedir)

        with torch.no_grad():
            cdata={}
            cdata['mu']=[]
            cdata['T']=[]
            for i, (input_, T) in enumerate(test_loader):

                #if(i>=test_loader.dataset.__len__()):
                if (i >= 800):
                    break

                output_,mu,logvar= self.G(input_)
                data={}
                data['input']=(torch.squeeze(input_.type(torch.FloatTensor))).numpy()
                data['input']=data['input'].astype(np.int8)
                data['output']=(torch.squeeze(output_.type(torch.FloatTensor).view(input_.size()))).numpy()
                data['output'] = (data['output'])
                data['T'] = (torch.squeeze(T.type(torch.FloatTensor))).numpy()
                data['T'] = (data['T'])
                cdata['T'].append(data['T'])
                data['mu']=(torch.squeeze(mu.type(torch.FloatTensor))).numpy()
                data['mu'] = (data['mu'])
                cdata['mu'].append(data['mu'])
                data['logvar']=(torch.squeeze(logvar.type(torch.FloatTensor))).numpy()
                data['logvar'] = (data['logvar'])
                scipy.io.savemat(self.save_path+savedir + "/" + str(i)+ ".mat",data)
                self.logger.will_write("[Save] fname: " +str(i))
            scipy.io.savemat(self.save_path + savedir + "/cdata.mat", cdata)
            self.logger.will_write("[Save]cdata ")
        print("End Test\n")

    def activationSave(self, img_path, savedir=None):
        print("\nStart activationSave")

        if savedir==None:
            savedir='/activation/test'
        else:
            savedir='/activation/'+savedir

        if os.path.exists(self.save_path+'/activation') is False:
            os.mkdir(self.save_path + '/activation')
        if os.path.exists(self.save_path+savedir) is False:
            os.mkdir(self.save_path+savedir)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = (torch.squeeze(output.detach()).type(torch.FloatTensor)).numpy()

            return hook

        data_pack = io.loadmat(img_path)
        input_np = data_pack['input']
        input = torch.from_numpy(input_np).view(1, 1, 128, 128, 64)
        input = input.to(torch.float)

        # print(model.module)
        self.G.module.layer1.register_forward_hook(get_activation('layer1'))
        self.G.module.layer2.register_forward_hook(get_activation('layer2'))
        self.G.module.layer3.register_forward_hook(get_activation('layer3'))
        self.G.module.layer4.register_forward_hook(get_activation('layer4'))
        self.G.module.ConBR1.register_forward_hook(get_activation('ConBR1'))
        self.G.module.ConBR2.register_forward_hook(get_activation('ConBR2'))
        self.G.module.ConBR3.register_forward_hook(get_activation('ConBR3'))
        output = self.G(input)
        scipy.io.savemat(self.save_path + savedir + "/1.mat", activation)
        print("\nEnd activationSave")

    def latent_sample(self, z_input, N):
        print("\nStart latent sample")
        temp = self.G.module.decode(torch.FloatTensor(z_input).to(self.torch_device)).reshape(-1, N * N).cpu().detach().numpy()
        #draws = 2*np.random.uniform(size=temp.shape)-1
        #samples = np.array(draws < temp).astype(int)

        samples = np.array(temp>0)
        return samples