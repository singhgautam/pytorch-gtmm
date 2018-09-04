"""Recall Task Parameters."""
import torch
import torchvision
from torchvision import datasets, transforms
from itertools import cycle
import os
from taskbase import TaskBaseParams
from variationalmodels.variationalmodel_convdeconv_digit import VariationalModelConvDeconvDigit

class ParityRecallTaskMNISTParams(TaskBaseParams):
    name = "parity-recall-task-mnist-dev"

    sequence_width = 28*28
    sequence_l = 10
    sequence_k = 5

    controller_size = 256
    controller_layers = 3

    memory_n = 128
    memory_m = 32
    num_read_heads = 1

    variational_hidden_size = 400

    clip_grad_thresh = 10

    num_batches = 1000000
    batch_size = 10

    rmsprop_lr = 1e-4
    rmsprop_momentum = 0.9
    rmsprop_alpha = 0.95

    adam_lr = 1e-4

    save_every = 100
    illustrate_every = 100

    variationalmodel = VariationalModelConvDeconvDigit

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

        # indices of ones and zeros
        idx_ones_train = [idx for idx, (data, label) in enumerate(train_dataset) if int(label.numpy()) == 1]
        idx_zeros_train = [idx for idx, (data, label) in enumerate(train_dataset) if int(label.numpy()) == 0]
        idx_ones_test = [idx for idx, (data, label) in enumerate(test_dataset) if int(label.numpy()) == 1]
        idx_zeros_test = [idx for idx, (data, label) in enumerate(test_dataset) if int(label.numpy()) == 0]

        # zeros and ones from the training set
        ones_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(idx_ones_train)
        )
        self.ones_loader_train_iter = cycle(ones_loader_train)

        zeros_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(idx_zeros_train)
        )
        self.zeros_loader_train_iter = cycle(zeros_loader_train)

        # zeros and ones from the test set
        ones_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(idx_ones_test)
        )
        self.ones_loader_test_iter = cycle(ones_loader_test)

        zeros_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(idx_zeros_test)
        )
        self.zeros_loader_test_iter = cycle(zeros_loader_test)


    def labels_to_parity_img_batch(self, labels, train = True, device='cpu'):
        """
        Returns a tensor containing a batch of parity labels in the form of MNIST images
        :param labels: a list of labels
        """

        # initialize a parity images batch
        parity_img_batch = torch.zeros(len(labels), 28, 28, device=device)

        # load ones and zeros from the training or the test data sets as specified
        ones_loader_iter = self.ones_loader_train_iter
        zeros_loader_iter = self.zeros_loader_train_iter
        if not train:
            ones_loader_iter = self.ones_loader_test_iter
            zeros_loader_iter = self.zeros_loader_test_iter


        for i, label in enumerate(labels):
            _data = None
            if label%2 == 0: # even
                _data, _ = ones_loader_iter.next()
            else:
                _data, _ = zeros_loader_iter.next()
            _data = _data.squeeze()
            _data = _data.view(28, 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            parity_img_batch[i] = _data

        return parity_img_batch

    def generate_random_batch(self, device='cpu', train = True):
        data_iter = self.train_loader_iter
        if not train:
            data_iter = self.test_loader_iter

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, self.batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, self.batch_size, self.sequence_width, device=device)
        for i in range(self.sequence_l):
            _data, labels = data_iter.next()
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            if i < self.sequence_k:
                parity_img_batch = self.labels_to_parity_img_batch(labels.numpy(), device=device)
                parity_img_batch = parity_img_batch.view(-1, 28*28)
                outp[i, :, :] = parity_img_batch

        return inp, outp

    def generate_illustrative_random_batch(self, device='cpu'):
        data_iter = self.illustration_loader_iter
        batch_size = 1
        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, batch_size, self.sequence_width, device=device)
        for i in range(self.sequence_l):
            _data, labels = data_iter.next()
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            if i < self.sequence_k:
                parity_img_batch = self.labels_to_parity_img_batch(labels.numpy(), train=False, device=device)
                parity_img_batch = parity_img_batch.view(-1, 28*28)
                outp[i, :, :] = parity_img_batch

        return inp, outp

    def create_images(self, batch_num, X, Y, Y_out, Y_out_binary, attention_history, modelcell):
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

        _Y = torch.cat([Y[i].view(28,28) for i in range(Y.size(0))], dim = 1)
        torchvision.utils.save_image(_Y.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y.png'.format(path, batch_num))

        _Y_out = torch.cat([Y_out[i].view(28,28) for i in range(Y_out.size(0))], dim = 1)
        torchvision.utils.save_image(_Y_out.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y-out.png'.format(path, batch_num))

        torchvision.utils.save_image(attention_history.cpu().detach(),
                                     '{}/batch-{}-attention.png'.format(path, batch_num))

        _Y_out_binary = torch.cat([Y_out_binary[i].view(28, 28) for i in range(Y_out_binary.size(0))], dim=1)
        torchvision.utils.save_image(_Y_out_binary.cpu().detach().squeeze(1),
                                     '{}/batch-{}-Y-out-binary.png'.format(path, batch_num))

        memory = modelcell.memory.memory.cpu().detach().squeeze(0)
        if memory.size()[1] == 1:
            memory = memory.repeat(1, 2)
        torchvision.utils.save_image(memory,
                                     '{}/batch-{}-mem.png'.format(path, batch_num))
