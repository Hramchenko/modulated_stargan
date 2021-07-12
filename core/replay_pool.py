import random
import torch

class ReplayPool():
    """
    This class implements a buffer that stores previously generated data.

    This buffer enables us to update discriminators using a history of generated data
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        self.poolSize = pool_size
        self.data = []

    def query(self, fake_data):
        assert(isinstance(fake_data, dict))
        if self.poolSize == 0:  # if the buffer size is 0, do nothing
            return fake_data
        result = []
        batch_size = None
        for k in fake_data:
            if batch_size is None:
                batch_size = fake_data[k].shape[0]
                continue
            assert(fake_data[k].shape[0] == batch_size)
        for idx in range(batch_size):
            rec = {}
            for k in fake_data:
                rec[k] = fake_data[k][idx]
            if len(self.data) < self.poolSize:
                self.data.append(rec)
                result.append(rec)
                continue
            if random.random() < 0.5:
                result.append(rec)
                continue
            random_idx = random.randint(0, len(self.data) - 1)
            random_rec = self.data[random_idx]
            self.data[random_idx] = rec
            result.append(random_rec)
        result_ = {}
        for k in fake_data:
            l = []
            for rec in result:
                l.append(rec[k])
            result_[k] = torch.stack(l, 0) 
        return result_
