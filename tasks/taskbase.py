class TaskBaseParams(object):
    def generate_random_batch(self, device, train):
        raise NotImplementedError

    def generate_illustrative_random_batch(self, device):
        raise NotImplementedError