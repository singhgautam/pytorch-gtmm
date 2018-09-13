"""Recall Task Parameters."""
import torch
import torchvision
from torchvision import datasets, transforms
from itertools import cycle
import os
from taskbase import TaskBaseParams
from variationalmodels.variationalmodel_convdeconv_digit import VariationalModelConvDeconvDigit
from samplers.mnistdigitsampler import MNISTDigitSampler
import numpy as np

class DynamicDependencyTaskMNISTParams(TaskBaseParams):
    name = "dynamic-dependency-task-mnist"

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
        assert self.sequence_k <= self.sequence_l, \
            "Meaningful sequence is larger than to the entire sequence length"
        assert self.sequence_l >= 10, \
            "Dynamic dependency task expects to have at least 10 memory locations in the input sequence."
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

        # per-digit data loaders
        self.train_digit_iterators = {}
        self.test_digit_iterators = {}

        for digit in range(10):
            _train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                sampler=MNISTDigitSampler(train_dataset, digit)
            )
            self.train_digit_iterators[digit] = cycle(_train_loader)

            _test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                sampler=MNISTDigitSampler(test_dataset, digit)
            )
            self.test_digit_iterators[digit] = cycle(_test_loader)

    def labels_to_img_batch(self, labels, train = True, device='cpu'):
        """
        Returns a tensor containing a batch of parity labels in the form of MNIST images
        :param labels: a list of labels
        """

        # initialize images batch
        img_batch = torch.zeros(len(labels), 28, 28, device=device)

        # route a digit iterator
        digit_iterators = self.train_digit_iterators
        if not train:
            digit_iterators = self.test_digit_iterators


        for i, label in enumerate(labels):
            _data, _ = digit_iterators[label].next()
            _data = _data.squeeze()
            _data = _data.view(28, 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            img_batch[i] = _data

        return img_batch

    def generate_random_batch(self, device='cpu', train = True):
        """
        Generates a random batch of input and output images for one iteration iteration
        :param device:
        :param train:
        :return: (inp, outp) where inp and outp are tensors of appropriate dimensions
        """
        data_iter = self.train_loader_iter
        if not train:
            data_iter = self.test_loader_iter

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, self.batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, self.batch_size, self.sequence_width, device=device)
        labels = []
        for i in range(self.sequence_l):
            _data, frame_labels = data_iter.next()
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            labels.append(frame_labels.numpy())
        labels = np.array(labels)

        out_labels = np.zeros([self.sequence_k, self.batch_size])
        for i_seq in range(self.batch_size):
            frame_labels = labels[:, i_seq]
            location = frame_labels[-1]
            for i_frame in range(self.sequence_k):
                out_labels[i_frame][i_seq] = frame_labels[location]
                location = frame_labels[location]

        for i_frame in range(self.sequence_k):
            img_batch = self.labels_to_img_batch(out_labels[i_frame], device=device)
            img_batch = img_batch.view(-1, 28*28)
            outp[i_frame, :, :] = img_batch

        return inp, outp

    def generate_illustrative_random_batch(self, device='cpu'):
        """
        Generates a random input sequence and its corresponding output for showing illustrations while training
        :param device:
        :return: (inp, outp) where inp and outp are tensors of appropriate dimensions
        """
        data_iter = self.illustration_loader_iter
        batch_size = 1
        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, batch_size, self.sequence_width, device=device)
        labels = []
        for i in range(self.sequence_l):
            _data, frame_labels = data_iter.next()
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            labels.append(frame_labels.numpy())
        labels = np.array(labels)

        out_labels = np.zeros([self.sequence_k, batch_size])
        for i_seq in range(batch_size):
            frame_labels = labels[:, i_seq]
            location = frame_labels[-1]
            for i_frame in range(self.sequence_k):
                out_labels[i_frame][i_seq] = frame_labels[location]
                location = frame_labels[location]

        for i_frame in range(self.sequence_k):
            img_batch = self.labels_to_img_batch(out_labels[i_frame], device=device)
            img_batch = img_batch.view(-1, 28*28)
            outp[i_frame, :, :] = img_batch

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
