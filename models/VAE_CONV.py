import torch
import torch.nn as nn
from torch.nn import functional as F


def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
    super(VAE, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=2),
        nn.ReLU(),
        Flatten()
    )

    self.fc1 = nn.Linear(h_dim, z_dim)
    self.fc2 = nn.Linear(h_dim, z_dim)
    self.fc3 = nn.Linear(z_dim, h_dim)

    self.decoder = nn.Sequential(
        UnFlatten(),
        nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
        nn.Sigmoid(),
    )



class VAECONV(nn.Module):

    def __init__(self, N=16, hidden_dim=512, z_dim=2):
        super(VAECONV, self).__init__()

        self.N=N
        self.input_dim=N*N
        self.output_dim=self.input_dim
        self.hidden_dim=hidden_dim
        self.z_dim=z_dim
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding =1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding =1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc21 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding = 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, bias=False)
        )

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1,self.hidden_dim)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.decoder(self.fc3(z).view(-1,32,4,4))
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
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