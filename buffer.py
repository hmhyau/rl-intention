import random
from collections import namedtuple
from utils import get_device, is_arraylike, as_list, mkdir_if_not_exist

import torch
import numpy as np
import cloudpickle as pickle

Transition = namedtuple('Transition',
                        ('observation', 'action', 'reward', 'observation_t1', 'done'))

class ReplayBuffer():
    def __init__(self, size, device=None, torch=True):
        self._storage = []
        self._maxsize = size
        self._idx = 0
        self.torch = torch
        self.device = device

    def __len__(self):
        return len(self._storage)

    @property
    def buffer_size(self):
        return self._maxsize

    def can_sample(self, sample_size):
        return len(self) >= sample_size

    def is_full(self):
        return len(self) == self._maxsize

    def add(self, obs, action, reward, obs_t1, done):
        data = self._convert_args((obs, action, reward, obs_t1, int(done)))
        # print(*data)
        data = Transition(*data)
        
        if(self._idx) >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._idx] = data
        self._idx = (self._idx + 1) % self._maxsize

    def _encode_sample(self, idxes, copy=False):
        obses, actions, rewards, obses_t1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, action, reward, obs_t1, done = data
            obses.append(np.array(obs, copy=copy))
            actions.append(np.array(action, copy=copy))
            rewards.append(np.array(reward, copy=copy))
            obses_t1.append(np.array(obs_t1, copy=copy))
            dones.append(np.array(done, copy=copy))

        data = Transition(obses, actions, rewards, obses_t1, dones)
        if self.torch:
            data = Transition(*tuple(map(self.to_torch, data)))

        return data
        
    def sample(self, batch_size, idx=False):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        if idx:
            return (idx, self._encode_sample(idxes))
        else:
            return self._encode_sample(idxes)

    def sample_from_idx(self, idxes):
        return self._encode_sample(idxes)

    def to_torch(self, array, copy=False):
        if copy:
            return torch.tensor(array).to(self.device)
        else:
            # print(array)
            # breakpoint()
            return torch.as_tensor(array).to(self.device)

    def save(self, save_path):
        full_path = save_path + '/buffer/'
        mkdir_if_not_exist(full_path)
        with open(full_path + 'buffer.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, load_path):
        full_path = load_path + '/buffer/'
        with open(full_path + 'buffer.pkl', 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.items())
            # for key, item in obj.items():
            #     try:
            #         self.__dict__[key] = item
            #     except KeyError:
            #         pass

    def _convert_args(self, data):
        return tuple(map(as_list, data))
            
if __name__ == '__main__':
    buffer = ReplayBuffer(1000)
    buffer.add([1, 2, 3], 1, 1, [2, 3, 4], 0)
    print(buffer.sample(1))
    buffer.save('./tmp/')

    new_buffer = ReplayBuffer(1000)
    new_buffer.load('./tmp/')
    print(new_buffer.sample(1))
