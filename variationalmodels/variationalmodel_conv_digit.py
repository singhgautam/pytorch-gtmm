import torch
import torch.nn as nn
import torch.nn.functional as F
from variationalmodelbase import VariationalModelBase


class VariationalModelConvDigit(VariationalModelBase):
    def init_nn_layers(self):

        x_dim = self.x_dim
        h_dim = self.h_dim
        z_dim = self.z_dim
        psi_dim = self.psi_dim

        # generator
        self.gen = nn.Sequential(
            nn.Linear(psi_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.gen_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self.gen_logvar = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()
        )

        # inference
        self.inf = ConvNetEncoder(psi_dim, h_dim)
        self.inf_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim)
        )
        self.inf_logvar = nn.Sequential(
            nn.Linear(h_dim, z_dim)
        )

        # prior
        self.prior = nn.Sequential(
            nn.Linear(psi_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim)
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(h_dim, z_dim)
        )

    def forward(self, psi_t, x_t, batch_size):
        # compute Q(Z|X) given context and x_t
        z_enc_mean, z_enc_logvar = self.inf(psi_t, x_t)
        z_mean = self.inf_mean(z_enc_mean)
        z_logvar = self.inf_logvar(z_enc_logvar)

        # compute prior over z_t given context
        z_enc_prior = self.prior(psi_t)
        z_mean_prior = self.prior_mean(z_enc_prior)
        z_logvar_prior = self.prior_logvar(z_enc_prior)

        # get a sample of z_t
        z_sample = self.sample_gaussian(z_mean, (0.5 * z_logvar).exp(), batch_size)

        # get distribution over x_t given z_t and the context
        _x_t_enc = self.gen(torch.cat([psi_t, z_sample], dim=1))
        _x_t_mean = self.gen_mean(_x_t_enc)

        # compute kld, log-likelihood of x_t and sub-elbo
        kld = self._kld_gauss(z_mean, z_logvar, z_mean_prior, z_logvar_prior)
        nll = self._nll_bernoulli(_x_t_mean, x_t)
        elbo_t = - nll - kld

        return z_sample, _x_t_mean, elbo_t

class ConvNetEncoder(nn.Module):
    def __init__(self, psi_dim, h_dim):
        super(ConvNetEncoder, self).__init__()

        self.psi_dim = psi_dim
        self.h_dim = h_dim

        self.bn1 = nn.BatchNorm2d(1)
        self.conv1x1_1 = nn.Conv2d(1, 8, kernel_size=1, padding=2)
        self.conv3x3_1 = nn.Conv2d(1, 8, kernel_size=3, padding=3)
        self.conv5x5_1 = nn.Conv2d(1, 8, kernel_size=5, padding=4)
        self.conv7x7_1 = nn.Conv2d(1, 8, kernel_size=7, padding=5)
        self.conv_dim_halving_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv1x1_2 = nn.Conv2d(32, 8, kernel_size=1, padding=0)
        self.conv3x3_2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv5x5_2 = nn.Conv2d(32, 8, kernel_size=5, padding=2)
        self.conv7x7_2 = nn.Conv2d(32, 8, kernel_size=7, padding=3)
        self.conv_dim_halving_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(8 * 8 * 64, 64)
        self.fc2_mean = nn.Linear(32 + psi_dim, h_dim)
        self.fc2_logvar = nn.Linear(32 + psi_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, h_dim)
        self.fc3_logvar = nn.Linear(h_dim, h_dim)



    def forward(self, psi, x):
        x = x.view(-1, 1, 28, 28)

        x = self.bn1(x)
        x = F.relu(x)
        x = torch.cat([
            self.conv1x1_1(x),
            self.conv3x3_1(x),
            self.conv5x5_1(x),
            self.conv7x7_1(x)
        ], dim=1)
        x = self.conv_dim_halving_1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.cat([
            self.conv1x1_2(x),
            self.conv3x3_2(x),
            self.conv5x5_2(x),
            self.conv7x7_2(x)
        ], dim=1)
        x = self.conv_dim_halving_2(x)

        x = x.view(-1, 8 * 8 * 64)

        x = self.fc1(x)
        x_mean = F.relu(self.fc2_mean(torch.cat([psi, x[:,:32]], dim = 1)))
        x_logvar = F.relu(self.fc2_logvar(torch.cat([psi, x[:,32:]], dim = 1)))

        # x_mean = F.relu(self.fc3_mean(x_mean))
        # x_logvar = F.relu(self.fc3_logvar(x_logvar))

        return x_mean, x_logvar