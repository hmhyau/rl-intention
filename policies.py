from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym.spaces import Discrete, Tuple, Box
from collections import defaultdict
from utils import convert_to_onehot, as_list, mkdir_if_not_exist
from layers import HNet, Reshape

BLACKJACK_OUTCOMES = ['Stick & Lose', 'Stick & Win', 'Hit & Lose', 'Stick & Draw', 'Hit Only']

class BasePolicy(ABC):
    def __init__(self, ob_space, ac_space, n_batch):
        # self.n_steps = n_steps
        self.n_batch = n_batch

        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        return isinstance(self.ac_space, Discrete)

    @property
    def predict(self, observation, deterministic=True):
        raise NotImplementedError

    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

class TabularPolicy(BasePolicy):    
    def __init__(self, ob_space, ac_space, n_batch=1, intent=False):
        super(TabularPolicy, self).__init__(ob_space, ac_space, n_batch)
        
        # OpenAI gym specific but can prob generalise better?
        if isinstance(self.ob_space, Tuple):
            self.n_obs = tuple(map(lambda x: x.n, self.ob_space))
            self.n_actions = self.ac_space.n

        if isinstance(self.ob_space, Discrete):
            self.n_obs = self.ob_space.n
            self.n_actions = self.ac_space.n

        self.intent = intent

        self._build()

    def predict(self, observation, deterministic=True):
        if deterministic:
            action = np.argmax(self.qvalues[observation])
            value = self.qvalues[observation, action]
        else:
            action = np.random.choice(self.n_actions)
            value = self.qvalues[observation, action]
        
        return action, value

    def _build(self):
        if isinstance(self.ob_space, Tuple):
            self.qvalues = np.zeros(shape=(self.n_obs + (self.n_actions,)))
        else:
            self.qvalues = np.zeros(shape=(self.n_obs, self.n_actions))
        if self.intent:
            self.hvalues = defaultdict(lambda: np.zeros(shape=self.qvalues.shape))

    def save(self, save_path):
        full_path = save_path + '/policy/'
        mkdir_if_not_exist(full_path)
        np.save(full_path + 'qtable.npy', self.qvalues)
        if self.intent:
            np.savez(full_path + 'htable.npz', dict(self.hvalues))

    def load(self, load_path):
        full_path = save_path + '/policy/'
        self.qvalues[:] = np.load(full_path + 'qtable.npy')
        with np.load(full_path + 'htable.npz', allow_pickle = True) as data:
            self.hvalues.update(data['arr_0'].item())

class BlackjackTabularPolicy(TabularPolicy):    
    def __init__(self, ob_space, ac_space, n_batch=1, intent=False):
        super(TabularPolicy, self).__init__(ob_space, ac_space, n_batch)
        
        # OpenAI gym specific but can prob generalise better?
        if isinstance(self.ob_space, Tuple):
            self.n_obs = tuple(map(lambda x: x.n, self.ob_space))
            self.n_actions = self.ac_space.n

        if isinstance(self.ob_space, Discrete):
            self.n_obs = self.ob_space.n
            self.n_actions = self.ac_space.n

        self.intent = intent

        self._build()

    def predict(self, observation, deterministic=True):
        if deterministic:
            action = np.argmax(self.qvalues[observation])
            value = self.qvalues[observation, action]
        else:
            action = np.random.choice(self.n_actions)
            # action = self.ac_space.sample()
            value = self.qvalues[observation, action]
        
        return action, value

    def _build(self):
        if isinstance(self.ob_space, Tuple):
            self.qvalues = np.zeros(shape=(self.n_obs + (self.n_actions,)))
        else:
            self.qvalues = np.zeros(shape=(self.n_obs, self.n_actions))
        if self.intent:
            self.hvalues = defaultdict(lambda: np.zeros(shape=self.qvalues.shape))
            self.intention = defaultdict(lambda: np.zeros(shape=len(BLACKJACK_OUTCOMES)))

    def save(self, save_path):
        full_path = save_path + '/policy/'
        mkdir_if_not_exist(full_path)
        np.save(full_path + 'qtable.npy', self.qvalues)
        if self.intent:
            np.savez(full_path + 'htable.npz', dict(self.hvalues))
            np.savez(full_path + 'intent.npz', dict(self.intention))

    def load(self, load_path):
        full_path = load_path + '/policy/'
        self.qvalues[:] = np.load(full_path + 'qtable.npy')
        # np.put(self.qvalues, np.load(full_path + 'qtable.npy'))
        with np.load(full_path + 'htable.npz', allow_pickle = True) as data:
            # print(defaultdict(tuple, data['arr_0'].item()))
            self.hvalues.update(data['arr_0'].item())
            # print(self.hvalues.keys())
        with np.load(full_path + 'intent.npz', allow_pickle=True) as data:
            # self.intention = data['arr_0'].item()
            self.intention.update(data['arr_0'].item())


class DeepPolicy(BasePolicy, nn.Module):
    def __init__(self, ob_space, ac_space, n_batch):
        nn.Module.__init__(self)
        BasePolicy.__init__(self, ob_space, ac_space, n_batch)

class DQNPolicy(DeepPolicy):
    def __init__(self, ob_space, ac_space, n_batch=16, layers=None,
                 feat_extraction="mlp", learning_rate=1e-4, double=False, dueling=False,
                 act_fun=nn.ReLU, optimizer_class=optim.Adam, optimizer_kwargs=None, device=None):
        super(DQNPolicy, self).__init__(ob_space, ac_space, n_batch)

        if layers is None:
            self.layers = [128, 512]
        else:
            self.layers = layers
        self.feat_extraction = feat_extraction
        self.lr = learning_rate
        self.double = double
        self.dueling = dueling
        self.act_fun = act_fun
        self.optimizer_class = optimizer_class
        self.device = device

        # OpenAI gym specific but can prob generalise better?
        if isinstance(self.ob_space, Tuple):
            self.n_obs = np.array(tuple(map(lambda x: x.n, self.ob_space)))
            self.obs_dim = self.n_obs.shape[0]
        elif isinstance(self.ob_space, Discrete):
            self.n_obs = self.ob_space.n
            self.obs_dim = 1
        elif isinstance(self.ob_space, Box):
            self.n_obs = self.ob_space.shape[0]
        else:
            self.n_obs = self.ob_space

        self.n_actions = ac_space.n

        self._build()

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            qvalues = self.forward(obs)
            if deterministic:
                action = qvalues.argmax(dim=1).reshape(-1).cpu().numpy()[0]
                value = qvalues.max(dim=1)[0].cpu().numpy()[0]
            else:
                action = np.random.choice(self.n_actions)
                value = qvalues[0][action]
        
        return action, value

    def forward(self, obs):
        if(obs.ndim == 1):
            # print(obs)
            obs = np.expand_dims(obs, axis=0)
            obs = torch.as_tensor(obs).float().to(self.device)
        return self.qnet(obs)

    def _make_qnet(self):
    # TBD: CNNs
        modules = []
        modules.append(nn.Linear(self.obs_dim, self.layers[0]))
        modules.append(self.act_fun())

        for idx in range(len(self.layers) - 1):
            modules.append(nn.Linear(self.layers[idx], self.layers[idx+1]))
            modules.append(self.act_fun())

        # Final layer
        modules.append(nn.Linear(self.layers[-1], self.n_actions))

        qnet = nn.Sequential(*modules).to(self.device)
        return qnet

    def _build(self):
        self.qnet = self._make_qnet()
        self.qnet_target = self._make_qnet()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = self.optimizer_class(self.qnet.parameters(), lr=self.lr)

    def save(self, save_path):
        full_path = save_path + '/policy/'
        mkdir_if_not_exist(full_path)
        torch.save(self.qnet_target.state_dict(), full_path + 'model.pth')
        torch.save(self.optimizer.state_dict(), full_path + 'optim.pth')

    def load(self, load_path):
        full_path = load_path + '/policy/'
        self.qnet.load_state_dict(torch.load(full_path + 'model.pth'))
        self.qnet_target.load_state_dict(torch.load(full_path + 'model.pth'))
        self.optimizer.load_state_dict(torch.load(full_path + 'optim.pth'))

class DBNPolicy(DeepPolicy):
    def __init__(self, ob_space, ac_space, n_batch=16, layers=None,
                 feat_extraction="mlp", learning_rate=1e-4, double=False, dueling=False,
                 act_fun=nn.ReLU, optimizer_class=optim.Adam, optimizer_kwargs=None, device=None):
        super(DBNPolicy, self).__init__(ob_space, ac_space, n_batch)

        if layers is None:
            self.layers = [128, 512]
        else:
            self.layers = layers

        self.feat_extraction = feat_extraction
        self.lr = learning_rate
        self.double = double
        self.dueling = dueling
        self.act_fun = act_fun
        self.optimizer_class = optimizer_class
        self.device = device

        # OpenAI gym specific but can prob generalise better?
        if isinstance(self.ob_space, Tuple):
            self.n_obs = np.array(tuple(map(lambda x: x.n, self.ob_space)))
            self.obs_dim = self.n_obs.shape[0]
        elif isinstance(self.ob_space, Discrete):
            self.n_obs = self.ob_space.n
            self.obs_dim = 1
        elif isinstance(self.ob_space, Box):
            self.n_obs = self.ob_space.shape[0]
        else:
            self.n_obs = self.ob_space

        self.n_actions = ac_space.n

        self.intent = intent
        self._build()

    def predict(self, obs, deterministic=True):
        # deterministic param has no use here since the prediction is dependent on Q-learning alg
        with torch.no_grad():
            bvalues = self.forward(obs)
        
        return bvalues

    def forward(self, obs):
        if(obs.ndim == 1):
            # print(obs)
            obs = np.expand_dims(obs, axis=0)
            obs = torch.as_tensor(obs).float().to(self.device)
        return self.hnet(obs)

    def _make_hnet(self):
        # deafault behaviour is to only predict map of states instead of state-action pair
        modules = []
        modules.append(nn.Linear(self.obs_dim, self.layers[0]))
        modules.append(self.act_fun())

        for idx in range(len(self.layers) - 1):
            modules.append(nn.Linear(self.layers[idx], self.layers[idx+1]))
            modules.append(self.act_fun())

        # Final layer
        modules.append(nn.Linear(self.layers[-1], self.n_obs))

        hnet = nn.Sequential(*modules).to(self.device)
        return hnet

    def _build(self):
        self.hnet = self._make_hnet()
        self.hnet_target = self._make_hnet()
        self.hnet_target.load_state_dict(self.hnet.state_dict())
        self.optimizer = self.optimizer_class(self.hnet.parameters(), lr=self.lr)

    def save(self, save_path):
        torch.save(self.hnet_target.state_dict(), save_path + '_model.pth')
        torch.save(self.optimizer.state_dict(), save_path + '_optim.pth')

    def load(self, load_path):
        self.hnet = torch.load(load_path + '_model.pth')
        self.hnet_target = torch.load(load_path + '_model.pth')
        self.optimizer = torch.load(load_path + 'optim.pth')

class IntentionPolicy(DeepPolicy):
    def __init__(self, ob_space, ac_space, ob_space_dbn=None, n_batch=16, layers_dqn=None, layers_dbn=None,
                 feat_extraction="mlp", learning_rate=1e-4, double=False, dueling=False, hnet="default",
                 act_fun=nn.ReLU, optimizer_class=optim.Adam, optimizer_kwargs=None, device=None):
        super(IntentionPolicy, self).__init__(ob_space, ac_space, n_batch)

        if layers_dqn is None:
            self.layers_dqn = [128, 512]
        else:
            self.layers_dqn = layers_dqn

        if layers_dbn is None:
            self.layers_dbn = [500, 2000]
        else:
            self.layers_dbn = layers_dbn

        self.feat_extraction = feat_extraction
        self.lr = learning_rate
        self.double = double
        self.dueling = dueling
        self.act_fun = act_fun
        self.optimizer_class = optimizer_class
        self.device = device

        # breakpoint()
        if isinstance(self.ob_space, Tuple):
            self.n_obs = np.array(tuple(map(lambda x: x.n, self.ob_space)))
            self.obs_dim = self.n_obs.shape[0]
        elif isinstance(self.ob_space, Discrete):
            self.n_obs = self.ob_space.n
            self.obs_dim = 1
        elif isinstance(self.ob_space, Box):
            self.n_obs = self.ob_space.shape[0]
        else:
            self.n_obs = self.ob_space

        # try to automatically deduce shape of DBN if arg is not provided
        if ob_space_dbn is None:
            if isinstance(self.ob_space, Tuple):
                self.n_obs_dbn = np.prod(tuple(map(lambda x: x.n, self.ob_space)))
            if isinstance(self.ob_space, Discrete):
                self.n_obs_dbn = self.ob_space.n
        else:
            self.n_obs_dbn = ob_space_dbn
        self.n_actions = ac_space.n

        # self.intent = intent
        self._build()

    def predict(self, obs, deterministic=True):
        # deterministic param has no use here since the prediction is dependent on Q-learning alg
        with torch.no_grad():
            # hvalues = self.forward(obs)
            # qvalues = self.forward(obs)
            qvalues, hvalues = self.forward(obs)
            # Reshape to shape of state space if observation space is a gym.Tuple
            if isinstance(self.ob_space, Tuple):
                hvalues = hvalues.view(-1, self.n_actions, *tuple(map(lambda x: x.n, self.ob_space)), self.n_actions)

            if deterministic:
                action = qvalues.argmax(dim=1).reshape(-1).cpu().numpy()[0]
                value = qvalues.max(dim=1)[0].cpu().numpy()[0]
            else:
                action = np.random.choice(self.n_actions)
                value = qvalues[0][action]

        return action, value, hvalues

    def forward(self, obs):
        if(obs.ndim == 1):
            # print(obs)
            obs = np.expand_dims(obs, axis=0)
        # obs = as_list(obs)
        obs = torch.as_tensor(obs).float().to(self.device)
        return self.qnet(obs), self.hnet(obs)


    def _make_hnet(self):
        # default behaviour is to only predict map of states instead of state-action pair
        modules = []
        modules.append(nn.Linear(self.obs_dim, self.layers_dbn[0]))
        modules.append(self.act_fun())

        for idx in range(len(self.layers_dbn)-1):
            modules.append(nn.Linear(self.layers_dbn[idx], self.layers_dbn[idx+1]))
            modules.append(self.act_fun())

        # Final layer
        if isinstance(self.n_obs_dbn, tuple):
            modules.append(nn.Linear(self.layers_dbn[-1], np.prod(self.n_obs_dbn)*self.n_actions**2))
            modules.append(Reshape(-1, self.n_actions, np.prod(self.n_obs_dbn), self.n_actions))
        else:
            modules.append(nn.Linear(self.layers_dbn[-1], self.n_obs_dbn*self.n_actions**2))
            modules.append(Reshape(-1, self.n_actions, self.n_obs_dbn, self.n_actions))

        hnet = nn.Sequential(*modules).to(self.device)
        return hnet
    
    def _make_qnet(self):
        modules = []
        print(self.n_obs, self.layers_dqn)
        modules.append(nn.Linear(self.obs_dim, self.layers_dqn[0]))
        modules.append(self.act_fun())

        for idx in range(len(self.layers_dbn) - 1):
            modules.append(nn.Linear(self.layers_dqn[idx], self.layers_dqn[idx+1]))
            modules.append(self.act_fun())

        # Final layer
        modules.append(nn.Linear(self.layers_dqn[-1], self.n_actions))

        qnet = nn.Sequential(*modules).to(self.device)
        return qnet

    def _build(self):
        self.qnet = self._make_qnet()
        self.qnet_target = self._make_qnet()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer_q = self.optimizer_class(self.qnet.parameters(), lr=self.lr)

        self.hnet = self._make_hnet()
        self.hnet_target = self._make_hnet()
        self.hnet_target.load_state_dict(self.hnet.state_dict())
        self.optimizer_h = self.optimizer_class(self.hnet.parameters(), lr=self.lr)

    def save(self, save_path):
        full_path = save_path + '/policy/'
        mkdir_if_not_exist(full_path)
        torch.save(self.qnet_target.state_dict(), full_path + 'model_q.pth')
        torch.save(self.optimizer_q.state_dict(), full_path + 'optim_q.pth')
        torch.save(self.hnet_target.state_dict(), full_path + 'model_h.pth')
        torch.save(self.optimizer_h.state_dict(), full_path + 'optim_h.pth')

    def load(self, load_path):
        full_path = load_path + '/policy/'
        self.qnet.load_state_dict(torch.load(full_path + 'model_q.pth'))
        self.qnet_target.load_state_dict(torch.load(full_path + 'model_q.pth'))
        self.optimizer_q.load_state_dict(torch.load(full_path + 'optim_q.pth'))

        self.hnet.load_state_dict(torch.load(full_path + 'model_h.pth'))
        self.hnet_target.load_state_dict(torch.load(full_path + 'model_h.pth'))
        self.optimizer_h.load_state_dict(torch.load(full_path + 'optim_h.pth'))

class IntentionAblatedPolicy(DeepPolicy):
    def __init__(self, ob_space, ac_space, ob_space_dbn=None, n_batch=16, layers_dqn=None, layers_dbn=None,
                 feat_extraction="mlp", learning_rate=1e-4, double=False, dueling=False, hnet="default",
                 act_fun=nn.ReLU, optimizer_class=optim.Adam, optimizer_kwargs=None, device=None):
        super(IntentionAblatedPolicy, self).__init__(ob_space, ac_space, n_batch)

        if layers_dqn is None:
            self.layers_dqn = [128, 512]
        else:
            self.layers_dqn = layers_dqn

        if layers_dbn is None:
            self.layers_dbn = [500, 2000]
        else:
            self.layers_dbn = layers_dbn

        self.feat_extraction = feat_extraction
        self.lr = learning_rate
        self.double = double
        self.dueling = dueling
        self.act_fun = act_fun
        self.optimizer_class = optimizer_class
        self.device = device

        # OpenAI gym specific but can prob generalise better?
        if isinstance(self.ob_space, Tuple):
            self.n_obs = np.array(tuple(map(lambda x: x.n, self.ob_space)))
            self.obs_dim = self.n_obs.shape[0]
        elif isinstance(self.ob_space, Discrete):
            self.n_obs = self.ob_space.n
            self.obs_dim = 1
        elif isinstance(self.ob_space, Box):
            self.n_obs = self.ob_space.shape[0]
        else:
            self.n_obs = self.ob_space

        # try to automatically deduce shape of DBN if arg is not provided
        if ob_space_dbn is None:
            if isinstance(self.ob_space, Tuple):
                self.n_obs_dbn = np.prod(tuple(map(lambda x: x.n, self.ob_space)))
            if isinstance(self.ob_space, Discrete):
                self.n_obs_dbn = self.ob_space.n
        else:
            self.n_obs_dbn = ob_space_dbn
        self.n_actions = ac_space.n

        # self.intent = intent
        self._build()

    def predict(self, obs, deterministic=True):
        # deterministic param has no use here since the prediction is dependent on Q-learning alg
        if(obs.ndim == 1):
            obs = np.expand_dims(obs, axis=0)
        obs = torch.as_tensor(obs).float().to(self.device)

        with torch.no_grad():
            # hvalues = self.forward(obs)
            # qvalues = self.forward(obs)
            # qvalues, hvalues = self.forward(obs)
            qvalues = self.qnet(obs)

            if deterministic:
                action = qvalues.argmax(dim=1).reshape(-1).cpu().numpy()[0]
                value = qvalues.max(dim=1)[0].cpu().numpy()[0]
            else:
                action = np.random.choice(self.n_actions)
                value = qvalues[0][action]

            # action = torch.as_tensor(as_list(action)).long().to(self.device)
            # breakpoint()
            hvalues = self.hnet(obs, torch.LongTensor([action]).to(self.device)[None, ...])
        return action, value, hvalues

    def forward(self, obs):
        if(obs.ndim == 1):
            # print(obs)
            obs = np.expand_dims(obs, axis=0)
        # obs = as_list(obs)
        obs = torch.as_tensor(obs).float().to(self.device)
        return self.qnet(obs), self.hnet(obs, act)


    def _make_hnet(self):
        # default behaviour is to only predict map of states instead of state-action pair
        # modules = []
        # modules.append(nn.Linear(self.n_obs, self.layers_dbn[0]))
        # modules.append(self.act_fun())

        # for idx in range(len(self.layers_dbn)-1):
        #     modules.append(nn.Linear(self.layers_dbn[idx], self.layers_dbn[idx+1]))
        #     modules.append(self.act_fun())

        # # Final layer
        # modules.append(nn.Linear(self.layers_dbn[-1], self.n_obs_dbn * self.n_actions * self.n_actions))
        # modules.append(Reshape(-1, self.n_actions, self.n_obs_dbn, self.n_actions))

        # hnet = nn.Sequential(*modules).to(self.device)
        hnet = HNet(self.obs_dim, self.n_actions, self.n_obs_dbn).to(self.device)
        return hnet
    
    def _make_qnet(self):
        modules = []
        
        modules.append(nn.Linear(self.obs_dim, self.layers_dbn[0]))
        modules.append(self.act_fun())

        for idx in range(len(self.layers_dbn) - 1):
            modules.append(nn.Linear(self.layers_dbn[idx], self.layers_dbn[idx+1]))
            modules.append(self.act_fun())

        # Final layer
        modules.append(nn.Linear(self.layers_dbn[-1], self.n_actions))

        qnet = nn.Sequential(*modules).to(self.device)
        return qnet

    def _build(self):
        self.qnet = self._make_qnet()
        self.qnet_target = self._make_qnet()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer_q = self.optimizer_class(self.qnet.parameters(), lr=self.lr)

        self.hnet = self._make_hnet()
        self.hnet_target = self._make_hnet()
        self.hnet_target.load_state_dict(self.hnet.state_dict())
        print(self.hnet)
        self.optimizer_h = self.optimizer_class(self.hnet.parameters(), lr=self.lr)

    def save(self, save_path):
        full_path = save_path + '/policy/'
        mkdir_if_not_exist(full_path)
        torch.save(self.qnet_target.state_dict(), full_path + 'model_q.pth')
        torch.save(self.optimizer_q.state_dict(), full_path + 'optim_q.pth')
        torch.save(self.hnet_target.state_dict(), full_path + 'model_h.pth')
        torch.save(self.optimizer_h.state_dict(), full_path + 'optim_h.pth')

    def load(self, load_path):
        full_path = load_path + '/policy/'
        # breakpoint()
        self.qnet.load_state_dict(torch.load(full_path + 'model_q.pth'))
        self.qnet_target.load_state_dict(torch.load(full_path + 'model_q.pth'))
        self.optimizer_q.load_state_dict(torch.load(full_path + 'optim_q.pth'))

        self.hnet.load_state_dict(torch.load(full_path + 'model_h.pth'))
        self.hnet_target.load_state_dict(torch.load(full_path + 'model_h.pth'))
        self.optimizer_h.load_state_dict(torch.load(full_path + 'optim_h.pth'))