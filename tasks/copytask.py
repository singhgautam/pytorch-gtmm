"""Copy Task Parameters."""
from attr import attrs, attrib, Factory
from torch import nn
from torch import optim
import torch
import math


@attrs
class CopyTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_read_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=8, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=5, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=9, convert=int)
    clip_grad_thresh = attrib(default=5, convert=int)
    variational_hidden_size = attrib(default=100, convert=int)
    num_batches = attrib(default=100000, convert=int)
    batch_size = attrib(default=10, convert=int)
    rmsprop_lr = attrib(default=1e-5, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    save_every = attrib(default=100, convert=int)
    illustrate_every = attrib(default=100, convert=int)

    def get_illustrative_sample(self, device = 'cpu'):
        '''Sequence will represent a rectified sine wave'''
        seq_len = self.sequence_max_len
        seq = torch.zeros(seq_len, 1, self.sequence_width, device = device)
        for i in range(seq_len):
            seq[i,0,int(self.sequence_width*abs(math.sin(2*i*math.pi/seq_len)))%self.sequence_width] = 1.0
        inp = torch.zeros(seq_len + 1, 1, self.sequence_width + 1, device = device)
        inp[:seq_len, :, :self.sequence_width] = seq
        inp[seq_len, :, self.sequence_width] = 1.0  # delimiter in our control channel

        outp = torch.zeros(seq_len, 1, self.sequence_width + 1, device=device)
        outp[:seq_len, :, :self.sequence_width] = seq

        return inp, outp
