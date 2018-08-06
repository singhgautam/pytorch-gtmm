import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class VariationalModelBase(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, psi_dim):
        super(VariationalModelBase, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.psi_dim = psi_dim

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.to(self.device)

        self.init_nn_layers()

    def init_nn_layers(self):
        raise NotImplementedError

    def forward(self, psi_t, x_t, batch_size):

        # compute Q(Z|X) given context and x_t
        z_enc = self.inf(torch.cat([psi_t, x_t], dim=1))
        z_mean = self.inf_mean(z_enc)
        z_logvar = self.inf_logvar(z_enc)

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
        nll =  self._nll_bernoulli(_x_t_mean, x_t)
        elbo_t = - nll - kld

        return z_sample, _x_t_mean, elbo_t

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def sample_gaussian(self, mean, std, batch_size):
        normalsample = torch.randn(mean.size(), device = self.device)
        return normalsample * std + mean  # scale the vector based on std and add mean


    def sample_x_mean(self, psi, batch_size):
        # compute prior over z_t given context
        z_enc_prior = self.prior(psi)
        z_mean_prior = self.prior_mean(z_enc_prior)
        z_logvar_prior = self.prior_logvar(z_enc_prior)
        z_sample = self.sample_gaussian(z_mean_prior, (0.5 * z_logvar_prior).exp(), batch_size)
        x_enc = self.gen(torch.cat([psi, z_sample], dim=1))
        x_gen_mean = self.gen_mean(x_enc)
        return z_sample, x_gen_mean


    def _kld_gauss(self, mean_1, logvar_1, mean_2, logvar_2):
        """Using std to compute KLD"""

        kld_element = logvar_2 - logvar_1 + (logvar_1.exp() + (mean_1 - mean_2).pow(2)) / (logvar_2.exp() + 1e-8) - 1
        return 0.5 * torch.sum(kld_element, dim = 1)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta + 1e-8) + (1 - x) * torch.log(1 - theta + 1e-8), dim=1)
        # return F.binary_cross_entropy(theta, x, size_average=False)