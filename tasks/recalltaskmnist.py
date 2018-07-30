"""Recall Task Parameters."""
import torch
import torchvision
from torchvision import datasets, transforms
from itertools import cycle
import os
from taskbase import TaskBaseParams

class RecallTaskMNISTParams(TaskBaseParams):
    name = "recall-task-mnist"

    sequence_width = 28*28
    sequence_l = 10
    sequence_k = 5

    controller_size = 100
    controller_layers = 1

    memory_n = 128
    memory_m = 20
    num_read_heads = 1

    variational_hidden_size = 400

    clip_grad_thresh = 5

    num_batches = 100000
    batch_size = 10

    rmsprop_lr = 1e-4
    rmsprop_momentum = 0.9
    rmsprop_alpha = 0.95

    adam_lr = 1e-4

    save_every = 100
    illustrate_every = 100

    def __init__(self):
        assert self.sequence_k <= self.sequence_l, "Meaningful sequence is larger than to the entire sequence length"
        self.load_MNIST()

    def load_MNIST(self):
        # train data set
        train_dataset = datasets.MNIST('data',
                       train=True,
                       download=True,
                       transform=transforms.ToTensor())

        # test data set
        test_dataset = datasets.MNIST('data',
                       train=False,
                       transform=transforms.ToTensor())

        # train data generator
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )
        self.train_loader_iter = cycle(train_loader)

        # test data generator
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )
        self.test_loader_iter = cycle(test_loader)

        # illustration data generator
        illustration_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle = True
        )
        self.illustration_loader_iter = cycle(illustration_loader)


    def generate_random_batch(self, device='cpu', train = True):
        data_iter = self.train_loader_iter
        if not train:
            data_iter = self.test_loader_iter

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, self.batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, self.batch_size, self.sequence_width, device=device)
        for i in range(self.sequence_l):
            _data, _ = data_iter.next()
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            if i < self.sequence_k:
                outp[i, :, :] = _data

        return inp, outp

    def generate_illustrative_random_batch(self, device='cpu'):
        data_iter = self.illustration_loader_iter
        batch_size = 1
        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, batch_size, self.sequence_width, device=device)
        for i in range(self.sequence_l):
            _data, _ = data_iter.next()
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            if i < self.sequence_k:
                outp[i, :, :] = _data

        return inp, outp

    def create_images(self, batch_num, X, Y_out, Y_out_binary, attention_history, modelcell):
        # make directories
        path = 'imsaves/{}'.format(self.name)
        try:
            os.makedirs(path)
        except:
            pass

        # save images
        _X = torch.cat([X[i].view(28,28) for i in range(X.size(0))], dim = 1)
        torchvision.utils.save_image(_X.cpu().detach().squeeze(1),
                                     '{}/batch-{}-X.png'.format(path, batch_num))

        _Y_out = torch.cat([Y_out[i].view(28,28) for i in range(Y_out.size(0))], dim = 1)
        torchvision.utils.save_image(_Y_out.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y.png'.format(path, batch_num))

        torchvision.utils.save_image(attention_history.cpu().detach(),
                                     '{}/batch-{}-attention.png'.format(path, batch_num))

        _Y_out_binary = torch.cat([Y_out_binary[i].view(28, 28) for i in range(Y_out_binary.size(0))], dim=1)
        torchvision.utils.save_image(_Y_out_binary.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y-binary.png'.format(path, batch_num))

        torchvision.utils.save_image(modelcell.memory.memory.cpu().detach().squeeze(0),
                                     '{}/batch-{}-mem.png'.format(path, batch_num))
