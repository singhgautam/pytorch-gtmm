import torch
from torch import nn
from controller import LSTMController
from rom import ROM
from state import State
import numpy as np

def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

class ModelCell(nn.Module):
    def __init__(self, params):
        super(ModelCell, self).__init__()

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        # set params
        self.params = params

        # create memory
        self.memory = ROM(params.memory_n, params.memory_m)

        # create controller
        self.controller = LSTMController(self.memory.M,
                                         params.controller_size,
                                         params.controller_layers)

        # create state
        self.state = State(self.memory, self.controller, self.params)

        # create variational model
        self.vmodel = params.variationalmodel(params.sequence_width,
                                       params.variational_hidden_size,
                                       params.memory_m,
                                       params.memory_m * self.params.num_read_heads)

        # create FC layer for addressing using controller output
        self.addressing_params_sizes = [self.memory.M, 1, 1, 3, 1]
        self.head_address_sizes = [sum(self.addressing_params_sizes)] * self.params.num_read_heads
        self.fc1 =  nn.Linear(params.controller_size, sum(self.addressing_params_sizes) * self.params.num_read_heads)

        self.to(self.device)

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def forward(self, X, batch_size):
        cout, self.state.controllerstate.state = self.controller(self.state.latentstate.state,
                                                                 self.state.controllerstate.state)
        packed_address_params = self.fc1(cout)
        for head_idx, address_params in enumerate(_split_cols(packed_address_params, self.head_address_sizes)):
            k, beta, g, s, gamma = _split_cols(address_params, self.addressing_params_sizes)
            self.state.readstate[head_idx].w = self.memory.address(k, beta, g, s, gamma, self.state.readstate[head_idx].w)
            self.state.readstate[head_idx].r = self.memory.read(self.state.readstate[head_idx].w)
        self.state.latentstate.state, X_gen_mean, _elbo = self.vmodel(
            torch.cat([self.state.readstate[i].r for i in range(self.params.num_read_heads)], dim=1),
            X,
            batch_size)
        self.memory.write(self.state.latentstate.state)

        return _elbo, X_gen_mean

    def generate(self, batch_size):
        cout, self.state.controllerstate.state = self.controller(self.state.latentstate.state,
                                                                 self.state.controllerstate.state)
        packed_address_params = self.fc1(cout)
        for head_idx, address_params in enumerate(_split_cols(packed_address_params, self.head_address_sizes)):
            k, beta, g, s, gamma = _split_cols(address_params, self.addressing_params_sizes)
            self.state.readstate[head_idx].w = self.memory.address(k, beta, g, s, gamma, self.state.readstate[head_idx].w)
            self.state.readstate[head_idx].r = self.memory.read(self.state.readstate[head_idx].w)
        self.state.latentstate.state, X_gen_mean = self.vmodel.sample_x_mean(
            torch.cat([self.state.readstate[i].r for i in range(self.params.num_read_heads)], dim=1),
            batch_size)
        self.memory.write(self.state.latentstate.state)

        return X_gen_mean