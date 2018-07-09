"""An read-only memory implementation."""
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torch import nn
import numpy as np


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


class ROM(nn.Module):
    """Memory."""
    def __init__(self, N, M):
        """Initialize the Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(ROM, self).__init__()

        self.N = N
        self.M = M

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

    def reset(self, batch_size):
        """Reset the memory"""
        self.batch_size = batch_size
        self.write_loc = 0

        stdev = 1 / (np.sqrt(self.N + self.M))
        self.memory = torch.randn(self.N, self.M, device = self.device).repeat(batch_size, 1, 1) * stdev
        self.memory.abs_()

    def visualize(self, savefile):
        torchvision.utils.save_image(self.memory, savefile)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, a):
        if a is None:
            return
        # write
        new_memory = self.memory.clone()
        new_memory[:,self.write_loc,:a.size(1)] = a
        self.memory = new_memory
        # increment the head one step
        self.write_loc = (self.write_loc + 1) % self.N


    def address(self, k, beta, g, s, gamma, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param beta: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param gamma: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """

        # k = k.clone()
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)
        # Content focus
        wc = self._similarity(k, beta)
        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w_hat = self._shift(wg, s)
        w = self._sharpen(w_hat, gamma)

        return w

    def _similarity(self, k, beta):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size(), device = self.device)
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, w_hat, gamma):
        w = w_hat ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
