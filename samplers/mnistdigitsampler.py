from torch.utils.data.sampler import Sampler

class MNISTDigitSampler(Sampler):
    """
    Samples MNIST images based on the digit value
    Arguments:
        data_source (Dataset) : data set to sample from
    """
    def __init__(self, data_source, digit):
        self.data_source = data_source
        self.indices = (idx for idx, (data, label) in enumerate(data_source) if int(label.numpy()) == digit)

    def __iter__(self):
        return self.indices

    def __len__(self, val):
        return len(self.idx_dict[val])