import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEFC(nn.Module):

    def __init__(self, N=16, hidden_dim=100, z_dim=2):
        super(VAEFC, self).__init__()

        self.N=N
        self.input_dim=N*N
        self.output_dim=self.input_dim
        self.hidden_dim=hidden_dim
        self.z_dim=z_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.output_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1,self.input_dim)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAEFC_2(nn.Module):

    def __init__(self, N=16, hidden_dim=100, z_dim=2):
        super(VAEFC_2, self).__init__()

        self.N=N
        self.input_dim=N*N
        self.output_dim=self.input_dim
        self.hidden_dim=hidden_dim
        self.z_dim=z_dim

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim)
                                 )
        self.fc21 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc3 = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.hidden_dim)
                      )
        self.fc4 = nn.Linear(self.hidden_dim, self.output_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1,self.input_dim)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEFC_CONV(nn.Module):

    def __init__(self, N=16, hidden_dim=100, z_dim=2):
        super(VAEFC_2, self).__init__()

        self.N=N
        self.input_dim=N*N
        self.output_dim=self.input_dim
        self.hidden_dim=hidden_dim
        self.z_dim=z_dim

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim)
                                 )
        self.fc21 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc3 = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                      nn.ReLU(),
                      nn.Linear(self.hidden_dim, self.hidden_dim)
                      )
        self.fc4 = nn.Linear(self.hidden_dim, self.output_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1,self.input_dim)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    input2D = torch.Tensor(1, 1, 128, 112, 80)
    
    print("input shape : \t", input2D.shape)