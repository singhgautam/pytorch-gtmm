import torch
import torch.nn as nn


class VariationalModel(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, psi_dim):
        super(VariationalModel, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.psi_dim = psi_dim

        # generator
        self.gen = nn.Sequential(
            nn.Linear(psi_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.gen_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        self.gen_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        # inference
        self.inf = nn.Sequential(
            nn.Linear(psi_dim + x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.inf_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Sigmoid())
        self.inf_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(psi_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Sigmoid())
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.to(self.device)

    def forward(self, psi_t, x_t, batch_size):

        # compute Q(Z|X) given context and x_t
        z_enc = self.inf(torch.cat([psi_t, x_t], dim=1))
        z_mean = self.inf_mean(z_enc)
        z_std = self.inf_std(z_enc)

        # compute prior over z_t given context
        z_enc_prior = self.prior(psi_t)
        z_mean_prior = self.prior_mean(z_enc_prior)
        z_std_prior = self.prior_std(z_enc_prior)

        # get a sample of z_t
        z_sample = self.sample_gaussian(z_mean, z_std, batch_size)

        # get distribution over x_t given z_t and the context
        _x_t_enc = self.gen(torch.cat([psi_t, z_sample], dim=1))
        _x_t_mean = self.gen_mean(_x_t_enc)
        _x_t_std = self.gen_mean(_x_t_enc)

        # compute kld, log-likelihood of x_t and sub-elbo
        kld = self._kld_gauss(z_mean, z_std, z_mean_prior, z_std_prior)
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
        z_std_prior = self.prior_std(z_enc_prior)
        z_sample = self.sample_gaussian(z_mean_prior, z_std_prior, batch_size)
        x_enc = self.gen(torch.cat([psi, z_sample], dim=1))
        x_gen_mean = self.gen_mean(x_enc)
        return z_sample, x_gen_mean


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element, dim = 1)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta), dim=1)


    def _nll_gauss(self, mean, std, x):
        pass