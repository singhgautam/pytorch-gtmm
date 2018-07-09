import torch
import torch.nn as nn


class VariationalModel(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, psi_dim, num_montecarlo = 1):
        super(VariationalModel, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.psi_dim = psi_dim
        self.num_montecarlo = num_montecarlo

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

        # compute Q(Z|X)
        z_enc = self.inf(torch.cat([psi_t, x_t], dim=1))
        z_mean = self.inf_mean(z_enc)
        z_std = self.inf_std(z_enc)

        # compute prior
        z_enc_prior = self.prior(psi_t)
        z_mean_prior = self.prior_mean(z_enc_prior)
        z_std_prior = self.prior_std(z_enc_prior)

        kld = self._kld_gauss(z_mean, z_std, z_mean_prior, z_std_prior)
        nll = 0
        for i in range(self.num_montecarlo):
            # sample Z using the inference model
            z_sample = self.sample_gaussian(z_mean, z_std)




    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def sample_gaussian(self, mean, std):
        normalsample = torch.randn(mean.size()).to(self.device)  # mean is a zero vector, all diagonals are 1 in std
        return normalsample.mul(std) + mean  # scale the vector based on std


    def sample_x_mean(self):
        z_sample = self.sample_gaussian(self.prior_z_mean, self.prior_z_std)
        z_enc = self.gen(z_sample)
        x_gen = self.gen_mean(z_enc)
        return x_gen


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))


    def _nll_gauss(self, mean, std, x):
        pass