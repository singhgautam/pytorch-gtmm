import torch
import random
import numpy as np
from torch import optim
from modelcell import ModelCell
from tasks.copytask import CopyTaskParams
import logging
import json
import time
import torchvision


LOGGER = logging.getLogger(__name__)

# initialize CUDA
print 'torch.version',torch.__version__
print 'torch.cuda.is_available()',torch.cuda.is_available()
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(time.time())

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def generate_random_batch(params, batch_size = None, device = 'cpu'):
    if batch_size is None:
        batch_size = params.batch_size
    # All batches have the same sequence length
    seq_len = params.sequence_len
    seq = np.random.binomial(1, 0.1, (seq_len,
                                      batch_size,
                                      params.sequence_width))
    seq = torch.Tensor(seq, device = device)

    # The input includes an additional channel used for the delimiter
    inp = torch.zeros(seq_len, batch_size, params.sequence_width, device = device)
    inp[:seq_len, :, :params.sequence_width] = seq
    # inp[seq_len, :, params.sequence_width] = 1.0  # delimiter in our control channel

    outp = torch.zeros(seq_len, batch_size, params.sequence_width, device=device)
    outp[:seq_len, :, :params.sequence_width] = seq

    return inp, outp

def clip_grads(model, range):
    """Gradient clipping to the range."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-range, range)

def batch_progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num - 1.0) % report_interval) + 1.0) / report_interval
    fill = int(progress * 40)
    print "\r\tBATCH [{}{}]: {} (ELBO: {:.4f})".format(
        "=" * fill,
        " " * (40 - fill),
        batch_num,
        last_loss)

def mean_progress(batch_num, mean_loss):
    print "BATCH {} (Mean ELBO: {:.4f})".format(
        batch_num,
        mean_loss)


"""MAKE MODEL"""

init_seed(1000)

params = CopyTaskParams()

# init model cell
modelcell = ModelCell(params)
modelcell.memory.reset(params.batch_size)
modelcell.state.reset(params.batch_size)
modelcell.controller.reset_parameters()

print 'modelcell.device {}'.format(modelcell.device)
print 'modelcell.memory.memory.device {}'.format(modelcell.memory.memory.device)
print 'modelcell.controller.device {}'.format(modelcell.controller.device)
print 'modelcell.state.readstate.w.device {}'.format(modelcell.state.readstate.w.device)
print 'modelcell.state.readstate.r.device {}'.format(modelcell.state.readstate.r.device)
print 'modelcell.state.controllerstate.device {}'.format(modelcell.state.controllerstate.device)
print 'modelcell.state.latentstate.state.device {}'.format(modelcell.state.latentstate.state.device)

# optimizer = optim.RMSprop(modelcell.parameters(),
#                           momentum=params.rmsprop_momentum,
#                           alpha=params.rmsprop_alpha,
#                           lr=params.rmsprop_lr)
optimizer = torch.optim.Adam(modelcell.parameters(), lr=params.adam_lr)

"""START TRAINING MODEL"""
loss_history = []
for batch_num in range(params.num_batches):
    # reset the states
    modelcell.memory.reset(params.batch_size)
    modelcell.state.reset(params.batch_size)

    # init optimizer
    optimizer.zero_grad()

    # generate data for the copy task
    X, Y = generate_random_batch(params, device = device)

    # input phase
    for i in range(X.size(0)):
        _elbo, _ = modelcell(X[i], params.batch_size)

    # output phase
    elbo = 0
    for i in range(Y.size(0)):
        _elbo, _ = modelcell(Y[i], params.batch_size)
        elbo += _elbo

    # mean ELBO for the entire batch
    mean_neg_elbo = -elbo.mean()
    mean_neg_elbo.backward()

    # log elbo history
    loss_history.append(mean_neg_elbo)

    clip_grads(modelcell, params.clip_grad_thresh)
    optimizer.step()

    batch_progress_bar(batch_num + 1, params.num_batches, last_loss=mean_neg_elbo)
    if batch_num % params.save_every == 0:
        mean_progress(batch_num,
                      sum(loss_history[-params.save_every:]) / params.save_every)

    if batch_num % params.illustrate_every == 0 :
        # X, Y = params.get_illustrative_sample(device=device)
        X, Y = generate_random_batch(params, batch_size=1, device=device)

        modelcell.memory.reset(batch_size=1)
        modelcell.state.reset(batch_size=1)

        attention_history = torch.zeros(X.size(0) + Y.size(0), modelcell.memory.N, device=device)

        # input phase
        for i in range(X.size(0)):
            _elbo, _ = modelcell(X[i], params.batch_size)
            attention_history[i] = modelcell.state.readstate.w.squeeze()

        # output phase
        Y_out = torch.zeros(Y.size(), device=device)
        for i in range(Y.size(0)):
            Y_out[i] = modelcell.generate(1)
            attention_history[X.size(0) + i] = modelcell.state.readstate.w.squeeze()

        Y_out_binary = Y_out.cpu().clone().data
        Y_out_binary.apply_(lambda x: 0 if x < 0.5 else 1)

        torchvision.utils.save_image(X.cpu().detach().squeeze(1), 'imsaves/illustrations/batch-{}-X.png'.format(batch_num))
        torchvision.utils.save_image(Y_out.cpu().detach().squeeze(1), 'imsaves/illustrations/batch-{}-Y.png'.format(batch_num))
        torchvision.utils.save_image(attention_history.cpu().detach(), 'imsaves/illustrations/batch-{}-attention.png'.format(batch_num))
        torchvision.utils.save_image(Y_out_binary.cpu().detach().squeeze(1), 'imsaves/illustrations/batch-{}-Y-binary.png'.format(batch_num))
        torchvision.utils.save_image(modelcell.memory.memory.cpu().detach().squeeze(0), 'imsaves/illustrations/batch-{}-mem.png'.format(batch_num))



