import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn
import numpy as np


class ReadState(nn.Module):
    def __init__(self, memory):
        super(ReadState, self).__init__()
        self.memory = memory

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

    def reset(self, batch_size):
        self.w = torch.zeros(batch_size, self.memory.N, device = self.device)
        self.w[:,0] = 1.0 # set reader attention at first spot in the memory
        self.r = self.memory.read(self.w)


class ControllerState(nn.Module):
    def __init__(self, controller):
        super(ControllerState, self).__init__()
        self.controller = controller

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        # starting hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.controller.num_layers, 1, self.controller.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.controller.num_layers, 1, self.controller.num_outputs) * 0.05)

        self.to(self.device)

    def reset(self, batch_size):
        h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        self.state = h, c


class LatentState(nn.Module):
    def __init__(self, latent_size):
        super(LatentState, self).__init__()

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.latent_size = latent_size

    def reset(self, batch_size):
        self.state = torch.zeros(batch_size, self.latent_size, device = self.device)

class State(nn.Module):
    def __init__(self, memory, controller):
        super(State, self).__init__()
        self.memory = memory
        self.controller = controller

    def reset(self, batch_size, num_read_states, latent_size):
        # setup readstate
        self.readstates = []
        for i in range(num_read_states):
            readstate = ReadState(self.memory)
            readstate.reset(batch_size)
            self.readstates.append(readstate)

        # setup controller state
        self.controllerstate = ControllerState(self.controller)
        self.controllerstate.reset(batch_size)

        # setup latent state
        self.latentstate = LatentState(latent_size)
        self.latentstate.reset(batch_size)
