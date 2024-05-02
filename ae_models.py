import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowAutoEncoder(nn.Module):
    def __init__(self, input_dim=28 * 28):
        super(ShallowAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder1 = nn.Linear(input_dim, 16)
        self.decoder1 = nn.Linear(16, input_dim)

    def encoder(self, x):
        return torch.tanh(self.encoder1(x))

    def decoder(self, z):
        return torch.sigmoid(self.decoder1(z))

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z)

    @staticmethod
    def loss_func(x_hat, x):
        return F.mse_loss(x_hat, x, reduction='sum')


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=28 * 28):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim

        # DNN as encoder
        self.encoder1 = nn.Linear(input_dim, 128)
        self.encoder2 = nn.Linear(128, 64)
        self.encoder3 = nn.Linear(64, 32)
        self.encoder4 = nn.Linear(32, 8)

        # DNN as decoder
        self.decoder1 = nn.Linear(8, 32)
        self.decoder2 = nn.Linear(32, 64)
        self.decoder3 = nn.Linear(64, 128)
        self.decoder4 = nn.Linear(128, input_dim)

    def encoder(self, x):
        h = torch.tanh(self.encoder1(x))
        h = torch.tanh(self.encoder2(h))
        h = torch.tanh(self.encoder3(h))
        return torch.tanh(self.encoder4(h))

    def decoder(self, z):
        h = torch.tanh(self.decoder1(z))
        h = torch.tanh(self.decoder2(h))
        h = torch.tanh(self.decoder3(h))
        return torch.sigmoid(self.decoder4(h))

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z)

    @staticmethod
    def loss_func(x_hat, x):
        return F.mse_loss(x_hat, x, reduction='sum')


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # CNN as encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 6)
        )

        # CNN as decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 6),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @staticmethod
    def loss_func(x_hat, x):
        return F.mse_loss(x_hat, x)


class VAE(nn.Module):
    def __init__(self, input_dim=28 * 28, latent_dim=8):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # DNN as encoder
        self.encoder1 = nn.Linear(input_dim, 128)
        self.mu = nn.Linear(128, 8)
        self.log_sigma2 = nn.Linear(128, 8)

        # DNN as decoder
        self.decoder1 = nn.Linear(8, 128)
        self.decoder2 = nn.Linear(128, input_dim)

    def encoder(self, x):
        h = torch.tanh(self.encoder1(x))
        return self.mu(h), torch.sqrt(torch.exp(self.log_sigma2(h)))

    @staticmethod
    def sampling(mu, std):  # Re parameterization trick
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h = torch.tanh(self.decoder1(z))
        return torch.sigmoid(self.decoder2(h))

    def forward(self, x):
        mu, std = self.encoder(x.view(-1, self.input_dim))
        z = self.sampling(mu, std)
        return self.decoder(z), mu, std

    @staticmethod
    def loss_func(x_hat, x, mu, std):
        # define the reconstruction loss
        ERR = F.binary_cross_entropy(x_hat, x, reduction='sum')
        # define the KL divergence loss
        KLD = -0.5 * torch.sum(1 + torch.log(std ** 2) - mu ** 2 - std ** 2)
        return ERR + KLD, ERR, KLD
