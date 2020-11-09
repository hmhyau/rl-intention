import torch
import torch.nn as nn
import torch.functional as F

from utils import convert_to_onehot

def mlp():
    pass

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class HNet(nn.Module):
    def __init__(self, n_obs, n_act, out_obs):
        super(HNet, self).__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.out_obs = out_obs

        self.act_stream = nn.Sequential(nn.Linear(self.n_act, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 512),
                                        nn.ReLU())
        
        self.intent_stream = nn.Sequential(nn.Linear(self.n_obs, 1024),
                             nn.ReLU(),
                             nn.Linear(1024, 1024),
                             nn.ReLU())

        self.shared_layers = nn.Sequential(nn.Linear(512+1024, 2048),
                                           nn.ReLU(),
                                           nn.Linear(2048, self.out_obs*self.n_act))
        # self.intent_layers = [nn.Linear(n_obs, 1024),
        #                       nn.Linear(1024, 1024)]
        # self.shared_layers = [nn.Linear(512+1024, 2048),
        #                       nn.Linear(2048, self.out_obs)]


    def _onehot(self, arr):
        onehot = torch.cuda.FloatTensor(arr.size()[0], self.n_act)
        onehot.zero_()
        onehot.scatter_(1, arr, 1)
        return onehot

    def forward(self, x, y):
        y = self._onehot(y)
        # for layer in self.act_layers:
        #     y = F.relu(layer(y))
        
        # for layer in self.intent_layers:
        #     x = F.relu(layer(x))

        # x = torch.cat((x, y), dim=1)
        
        # for layer in self.shared_layers[:-1]:
        #     x = F.relu(layer(x))
        y = self.act_stream(y)
        x = self.intent_stream(x)
        x = torch.cat((x, y), dim=1)
        x = self.shared_layers(x)
        x = x.view(-1, self.out_obs, self.n_act)
        return x