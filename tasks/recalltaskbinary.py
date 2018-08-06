"""Recall Task Parameters."""
import torch
import math
import numpy as np
import torchvision
import os
from taskbase import TaskBaseParams
from variationalmodels.variationalmodel_2fc import VariationalModel2FC

class RecallTaskBinaryParams(TaskBaseParams):
    name = "recall-task-binary"
    controller_size = 100
    controller_layers = 1
    num_read_heads = 1
    sequence_width = 20
    sequence_len = 5
    memory_n = 128
    memory_m = 10
    clip_grad_thresh = 5
    variational_hidden_size = 200
    num_batches = 10000
    batch_size = 10
    rmsprop_lr = 1e-4
    rmsprop_momentum = 0.9
    rmsprop_alpha = 0.95
    adam_lr = 1e-4
    save_every = 100
    illustrate_every = 100
    variationalmodel = VariationalModel2FC

    def generate_random_batch(self, batch_size=None, device='cpu'):
        if batch_size == None:
            batch_size = self.batch_size

        # All batches have the same sequence length
        seq_len = self.sequence_len
        seq = np.random.binomial(1, 0.1, (seq_len,
                                          batch_size,
                                          self.sequence_width))
        seq = torch.Tensor(seq, device=device)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len, batch_size, self.sequence_width, device=device)
        inp[:seq_len, :, :self.sequence_width] = seq
        # inp[seq_len, :, params.sequence_width] = 1.0  # delimiter in our control channel

        outp = torch.zeros(seq_len, batch_size, self.sequence_width, device=device)
        outp[:seq_len, :, :self.sequence_width] = seq

        return inp, outp

    def generate_illustrative_random_batch(self, device = 'cpu'):
        return self.generate_random_batch(batch_size=1, device = device)

    def create_images(self, batch_num, X, Y, Y_out, Y_out_binary, attention_history, modelcell):
        # make directories
        path = 'imsaves/{}'.format(self.name)
        try:
            os.makedirs(path)
        except:
            pass

        # save images
        torchvision.utils.save_image(X.cpu().detach().squeeze(1),
                                     '{}/batch-{}-X.png'.format(path, batch_num))
        torchvision.utils.save_image(Y.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y.png'.format(path, batch_num))
        torchvision.utils.save_image(Y_out.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y-out.png'.format(path, batch_num))
        torchvision.utils.save_image(attention_history.cpu().detach(),
                                     '{}/batch-{}-attention.png'.format(path, batch_num))
        torchvision.utils.save_image(Y_out_binary.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y-out-binary.png'.format(path, batch_num))
        torchvision.utils.save_image(modelcell.memory.memory.cpu().detach().squeeze(0),
                                     '{}/batch-{}-mem.png'.format(path, batch_num))




