import torch
import torch.nn as nn
from torch.nn import functional as F


class GMVAEFC_DEEP(nn.Module):

    def __init__(self, N=16, hidden_dim=100, z_dim=2, k=2):
        super(GMVAEFC_DEEP, self).__init__()

        self.N=N
        self.input_dim=N*N
        self.output_dim=self.input_dim
        self.hidden_dim=hidden_dim
        self.z_dim=z_dim
        self.k=k
        self.torch_device=torch.device("cuda")


        self.decode1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.decode2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decode3 = nn.Linear(self.hidden_dim, self.output_dim)

        self.qy_graph1=nn.Linear(self.input_dim,self.hidden_dim)
        self.qy_graph1_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.qy_graph2=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.qy_graph3=nn.Linear(self.hidden_dim,self.k)

        self.qz_graph1=nn.Linear(self.input_dim,self.hidden_dim)
        self.qz_graph2=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.qz_graph2_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.qz_graph3=nn.Linear(self.hidden_dim,self.k* self.z_dim)
        self.qz_graph4=nn.Linear(self.hidden_dim,self.k* self.z_dim)

        self.zm_graph=torch.nn.Parameter(torch.randn(self.k,self.z_dim).to(self.torch_device).requires_grad_(True))
        self.zv_graph=torch.nn.Parameter(torch.randn(self.k,self.z_dim).to(self.torch_device).requires_grad_(True))


    def px_graph(self, batch_size):
        zm_graph=self.zm_graph.expand(batch_size,self.k,self.z_dim)
        zv_graph=F.softplus(self.zv_graph.expand(batch_size, self.k, self.z_dim))
        return zm_graph, zv_graph

    def qy_graph(self, x):
        h1=F.relu(self.qy_graph1(x))
        h1 = F.relu(self.qy_graph1_1(h1))
        h2=F.relu(self.qy_graph2(h1))
        qy_logit=self.qy_graph3(h2)
        qy=F.softmax(qy_logit, dim=1)
        return qy_logit, qy

    def qz_graph(self, x):
        h1=F.relu(self.qz_graph1(x))
        h2=F.relu(self.qz_graph2(h1))
        h2 = F.relu(self.qz_graph2_2(h2))
        zm=self.qz_graph3(h2)
        zv=F.softplus(self.qz_graph4(h2))
        z=self.reparameterize(zm,zv)
        return z, zm, zv

    def reparameterize(self, mu, var):
        logvar=torch.log(var)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.tanh(self.decode1(z))
        h3 = torch.tanh(self.decode2(h3))
        return torch.tanh(self.decode3(h3))

    def forward(self, x):
        x=x.view(-1,self.input_dim)
        qy_logit,qy=self.qy_graph(x)
        z, zm, zv, zm_p, zv_p=[torch.zeros(x.shape[0],self.k,self.z_dim).to(self.torch_device)]*5
        px=torch.zeros(x.shape[0],self.k,self.output_dim).to(self.torch_device)

        z_,zm_,zv_=self.qz_graph(x)
        z=z_.view(x.size(0),self.k, self.z_dim)
        zm = zm_.view(x.size(0), self.k, self.z_dim)
        zv = zv_.view(x.size(0), self.k, self.z_dim)
        zm_p,zv_p=self.px_graph(x.size(0))

        px=self.decode(z.view(-1,self.z_dim)).view(x.size(0),self.k,self.output_dim)

        return px,z,zm,zv,zm_p,zv_p,qy,qy_logit

    def Loss(self, x, px, z, zm, zv, zm_p, zv_p, qy, qy_logit):
        ReconLoss=torch.zeros(x.shape[0],self.k).to(self.torch_device)
        KLD=torch.zeros(x.shape[0], self.k).to(self.torch_device)
        NENT=self.NENT(qy)
        BNENT=self.BNENT(qy)
        CLUST=self.Clustering(zm_p,zv_p)

        for i in range(self.k):
            ReconLoss[:,i]=self.ReconLoss(x,px[:,i].clone())
            KLD[:,i]=self.KLD(z[:,i].clone(),zm[:,i].clone(),zv[:,i].clone(),zm_p[:,i].clone(),zv_p[:,i].clone())

        FLoss = torch.mean((10*ReconLoss + KLD) * qy)+(torch.mean(NENT)-50*BNENT)+10*CLUST

        return FLoss,ReconLoss, KLD, NENT, BNENT

    def ReconLoss(self, x, px):
        return torch.sum(-torch.log(1 - torch.abs(px - x.view(-1, self.N * self.N)) / 2),dim=1)

    def KLD(self, z, zm, zv, zm_p, zv_p):
        KLD = -0.5 * torch.mean(- torch.log(zv_p) + torch.log(zv) - (zv+torch.pow((zm-zm_p),2))/zv_p, dim=1)
        return KLD

    def NENT(self, qy):
        NENT=-torch.mean(qy*torch.log(qy),dim=1)
        return NENT

    def BNENT(self, qy):
        qy=qy.mean(dim=0)
        BNENT=-torch.mean(qy*torch.log(qy))
        return BNENT

    def Clustering(self,zm_p, zv_p):
        CLUST=0
        for i in range(self.k):
            for j in range(i+1, self.k):
                CLUST=CLUST+1/torch.mean(torch.abs(zm_p[0,i,:]-zm_p[0,j,:]))
        CLUST=CLUST
        return CLUST

if __name__ == "__main__":
    input2D = torch.Tensor(1, 1, 128, 112, 80)
    
    print("input shape : \t", input2D.shape)