import torch
import torch.nn as nn
import torch.nn.functional as F
from variationalmodelbase import VariationalModelBase


class VariationalModelConvDeconvOmniglot(VariationalModelBase):
    def init_nn_layers(self):

        x_dim = self.x_dim
        h_dim = self.h_dim
        z_dim = self.z_dim
        psi_dim = self.psi_dim

        # generator
        self.gen = ConvNetDecoder(psi_dim, h_dim, z_dim)

        # inference
        self.inf = ConvNetEncoder(psi_dim, h_dim, z_dim)

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
        z_mean, z_logvar = self.inf(psi_t, x_t)

        # compute prior over z_t given context
        z_enc_prior = self.prior(psi_t)
        z_mean_prior = self.prior_mean(z_enc_prior)
        z_logvar_prior = self.prior_logvar(z_enc_prior)

        # get a sample of z_t
        z_sample = self.sample_gaussian(z_mean, (0.5 * z_logvar).exp(), batch_size)

        # get distribution over x_t given z_t and the context
        _x_t_mean = self.gen(torch.cat([psi_t, z_sample], dim=1))

        # compute kld, log-likelihood of x_t and sub-elbo
        kld = self._kld_gauss(z_mean, z_logvar, z_mean_prior, z_logvar_prior)
        nll = self._nll_bernoulli(_x_t_mean, x_t)
        elbo_t = - nll - kld

        return z_sample, _x_t_mean, elbo_t

    def sample_x_mean(self, psi, batch_size):
        # compute prior over z_t given context
        z_enc_prior = self.prior(psi)
        z_mean_prior = self.prior_mean(z_enc_prior)
        z_logvar_prior = self.prior_logvar(z_enc_prior)
        z_sample = self.sample_gaussian(z_mean_prior, (0.5 * z_logvar_prior).exp(), batch_size)
        x_gen_mean = self.gen(torch.cat([psi, z_sample], dim=1))
        return z_sample, x_gen_mean

class ConvNetEncoder(nn.Module):
    def __init__(self, psi_dim, h_dim, z_dim):
        super(ConvNetEncoder, self).__init__()

        self.psi_dim = psi_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.bn1 = nn.BatchNorm2d(1)
        self.bn_conv1x1_1 = nn.BatchNorm2d(8)
        self.bn_conv3x3_1 = nn.BatchNorm2d(8)
        self.bn_conv5x5_1 = nn.BatchNorm2d(8)
        self.bn_conv7x7_1 = nn.BatchNorm2d(8)
        self.conv1x1_1 = nn.Conv2d(1, 8, kernel_size=1, padding=2)
        self.conv3x3_1 = nn.Conv2d(1, 8, kernel_size=3, padding=3)
        self.conv5x5_1 = nn.Conv2d(1, 8, kernel_size=5, padding=4)
        self.conv7x7_1 = nn.Conv2d(1, 8, kernel_size=7, padding=5)
        self.conv_dim_halving_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.bn_conv1x1_2 = nn.BatchNorm2d(8)
        self.bn_conv3x3_2 = nn.BatchNorm2d(8)
        self.bn_conv5x5_2 = nn.BatchNorm2d(8)
        self.bn_conv7x7_2 = nn.BatchNorm2d(8)
        self.conv1x1_2 = nn.Conv2d(32, 8, kernel_size=1, padding=0)
        self.conv3x3_2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv5x5_2 = nn.Conv2d(32, 8, kernel_size=5, padding=2)
        self.conv7x7_2 = nn.Conv2d(32, 8, kernel_size=7, padding=3)
        self.conv_dim_halving_2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.bn_conv1x1_3 = nn.BatchNorm2d(8)
        self.bn_conv3x3_3 = nn.BatchNorm2d(8)
        self.bn_conv5x5_3 = nn.BatchNorm2d(8)
        self.bn_conv7x7_3 = nn.BatchNorm2d(8)
        self.conv1x1_3 = nn.Conv2d(32, 8, kernel_size=1, padding=0)
        self.conv3x3_3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv5x5_3 = nn.Conv2d(32, 8, kernel_size=5, padding=2)
        self.conv7x7_3 = nn.Conv2d(32, 8, kernel_size=7, padding=3)
        self.conv_dim_halving_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(14 * 14 * 64, 2 * psi_dim)
        self.fc2_mean = nn.Linear( 2 * psi_dim, h_dim)
        self.fc2_logvar = nn.Linear( 2 * psi_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, z_dim)
        self.fc3_logvar = nn.Linear(h_dim, z_dim)



    def forward(self, psi, x):
        x = x.view(-1, 1, 105, 105)

        x = self.bn1(x)
        x = F.relu(x)

        x = torch.cat([
            F.relu(self.bn_conv1x1_1(self.conv1x1_1(x))),
            F.relu(self.bn_conv3x3_1(self.conv3x3_1(x))),
            F.relu(self.bn_conv5x5_1(self.conv5x5_1(x))),
            F.relu(self.bn_conv7x7_1(self.conv7x7_1(x)))
        ], dim=1)
        x = self.conv_dim_halving_1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.cat([
            F.relu(self.bn_conv1x1_2(self.conv1x1_2(x))),
            F.relu(self.bn_conv3x3_2(self.conv3x3_2(x))),
            F.relu(self.bn_conv5x5_2(self.conv5x5_2(x))),
            F.relu(self.bn_conv7x7_2(self.conv7x7_2(x)))
        ], dim=1)
        x = self.conv_dim_halving_2(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = torch.cat([
            F.relu(self.bn_conv1x1_3(self.conv1x1_3(x))),
            F.relu(self.bn_conv3x3_3(self.conv3x3_3(x))),
            F.relu(self.bn_conv5x5_3(self.conv5x5_3(x))),
            F.relu(self.bn_conv7x7_3(self.conv7x7_3(x)))
        ], dim=1)
        x = self.conv_dim_halving_3(x)


        x = x.view(-1, 14 * 14 * 64)

        x = self.fc1(x)
        x_mean = F.relu(self.fc2_mean(torch.cat([psi, x[:,:self.psi_dim]], dim = 1)))
        x_logvar = F.relu(self.fc2_logvar(torch.cat([psi, x[:,self.psi_dim:]], dim = 1)))

        x_mean = self.fc3_mean(x_mean)
        x_logvar = self.fc3_logvar(x_logvar)

        return x_mean, x_logvar

class ConvNetDecoder(nn.Module):
    def __init__(self, psi_dim, h_dim, z_dim):
        super(ConvNetDecoder, self).__init__()

        self.psi_dim = psi_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(psi_dim + z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 64 * 14 * 14)

        self.deconv_dim_doubling_1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)

        self.deconv1x1_1 = nn.ConvTranspose2d(8, 32, kernel_size=1, padding=0)
        self.deconv3x3_1 = nn.ConvTranspose2d(8, 32, kernel_size=3, padding=1)
        self.deconv5x5_1 = nn.ConvTranspose2d(8, 32, kernel_size=5, padding=2)
        self.deconv7x7_1 = nn.ConvTranspose2d(8, 32, kernel_size=7, padding=3)

        self.bn2 = nn.BatchNorm2d(32)

        self.deconv_dim_doubling_2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(32)

        self.deconv1x1_2 = nn.ConvTranspose2d(8, 32, kernel_size=1, padding=0)
        self.deconv3x3_2 = nn.ConvTranspose2d(8, 32, kernel_size=3, padding=1)
        self.deconv5x5_2 = nn.ConvTranspose2d(8, 32, kernel_size=5, padding=2)
        self.deconv7x7_2 = nn.ConvTranspose2d(8, 32, kernel_size=7, padding=3)

        self.bn4 = nn.BatchNorm2d(32)

        self.deconv_dim_doubling_3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.bn5 = nn.BatchNorm2d(32)


        self.deconv1x1_3 = nn.ConvTranspose2d(8, 1, kernel_size=1, padding=0)
        self.deconv3x3_3 = nn.ConvTranspose2d(8, 1, kernel_size=3, padding=1)
        self.deconv5x5_3 = nn.ConvTranspose2d(8, 1, kernel_size=5, padding=2)
        self.deconv7x7_3 = nn.ConvTranspose2d(8, 1, kernel_size=7, padding=3)




    def forward(self, psi_cat_z):

        x = F.relu(self.fc1(psi_cat_z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 14, 14)

        x = self.deconv_dim_doubling_1(x, output_size=(28,28))
        x = self.bn1(x)
        x = F.relu(x)

        x = self.deconv1x1_1(x[:,0:8,:,:]) + self.deconv3x3_1(x[:,8:16,:,:]) + self.deconv5x5_1(x[:,16:24,:,:]) + self.deconv7x7_1(x[:,24:32,:,:])
        x = self.bn2(x)
        x = F.relu(x)

        x = self.deconv_dim_doubling_2(x, output_size=(55,55))
        x = self.bn3(x)
        x = F.relu(x)

        x = self.deconv1x1_2(x[:,0:8,:,:]) + self.deconv3x3_2(x[:,8:16,:,:]) + self.deconv5x5_2(x[:,16:24,:,:]) + self.deconv7x7_2(x[:,24:32,:,:])
        x = self.bn4(x)
        x = F.relu(x)

        x = self.deconv_dim_doubling_3(x, output_size=(109,109))
        x = self.bn5(x)
        x = F.relu(x)

        x = self.deconv1x1_3(x[:,0:8,:,:]) + self.deconv3x3_3(x[:,8:16,:,:]) + self.deconv5x5_3(x[:,16:24,:,:]) + self.deconv7x7_3(x[:,24:32,:,:])
        x = F.sigmoid(x)

        x = x[:,0,2:107,2:107] #remove padding
        x = x.contiguous().view(-1,105*105) #unwrap the image

        return x





