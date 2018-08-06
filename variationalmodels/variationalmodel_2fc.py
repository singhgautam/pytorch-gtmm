import torch
import torch.nn as nn
from variationalmodelbase import VariationalModelBase


class VariationalModel2FC(VariationalModelBase):
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
        self.inf = nn.Sequential(
            nn.Linear(psi_dim + x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
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