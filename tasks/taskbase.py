class TaskBaseParams(object):
    def generate_random_batch(self, device, train):
        raise NotImplementedError

    def generate_illustrative_random_batch(self, device):
        raise NotImplementedError

    def create_images(self, batch_num, X, Y, Y_out, Y_out_binary, attention_history, modelcell):
        raise NotImplementedError