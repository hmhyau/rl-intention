from abc import ABC, abstractmethod
import numpy as np
from utils import set_global_seeds, get_device, get_default_args, mkdir_if_not_exist, CartpoleEncoder
from buffer import ReplayBuffer

import json
import zipfile

import cloudpickle as pickle

from datetime import datetime

class BaseRLModel(ABC):
    def __init__(self, policy, env, policy_kwargs=None, seed=None):
        
        self.policy = policy
        if policy_kwargs is None:
            self.policy_kwargs = {}
        else:
            self.policy_kwargs = policy_kwargs

        self.seed = seed
        self.observation_space = None
        self.action_space = None
        self.ep_done = 0
        self.elapsed_steps = 0
        self.episode_reward = None


        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.env = env

        if seed is not None:
            self.set_random_seed(seed)

        self.exec_time = datetime.now()
        self.exec_str = self.exec_time.strftime("%Y%m%d%H%M%S")

    def get_env(self):
        return self.env
    
    def set_env(self, env):
        if env is None and self.env is None:
            print("Loading model without an environment.")

        elif env is None:
            raise ValueError("Trying to replace current environment with None.")

    def _init_timesteps(self, reset=True):
        if reset:
            self.num_timesteps = 0

    def set_random_seed(self, seed):
        if seed is None:
            return

        set_global_seeds(seed)
        if self.env is not None:
            self.env.seed(seed)
            self.env.action_space.np_random.seed(seed)
        self.action_space.seed(seed)

    @abstractmethod
    def learn(self, callbacks, total_timesteps, log_interval):
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation, deterministic):
        raise NotImplementedError

    # @abstractmethod
    # @classmethod
    def load(self, load_path, env=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path, **kwargs):
        raise NotImplementedError

class TabularRLModel(BaseRLModel):
    def __init__(
        self,
        policy,
        env,
        gamma,
        learning_rate,
        buffer_size,
        exploration_type,
        exploration_frac,
        exploration_ep,
        exploration_initial_eps,
        exploration_final_eps,
        double_q,
        policy_kwargs,
        seed,
        intent
        ):

        super(TabularRLModel, self).__init__(
            policy=policy,
            env=env, 
            policy_kwargs=policy_kwargs,
            seed=seed)

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.exploration_type = exploration_type
        self.exploration_frac = exploration_frac
        self.exploration_ep = exploration_ep
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.double_q = double_q
        self.intent = intent
        # self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        # self.policy = policy(self.observation_space, self.action_space, intent=True)

        self.policy_kwargs = get_default_args(self.policy)
        self.policy_kwargs['ob_space'] = self.observation_space
        self.policy_kwargs['ac_space'] = self.action_space
        self.policy_kwargs['intent'] = self.intent

        if policy_kwargs is not None:
            for key, val in policy_kwargs.items():
                self.policy_kwargs[key] = val
        # self.policy_kwargs['transform_func'] = transform_func

        # if policy_kwargs is None:
        #     self.policy = policy(self.observation_space, self.action_space,
        #                          intent=True, device=self.device)
        # else:
        self.policy = policy(**self.policy_kwargs)

        if self.buffer_size is None:
            self.replay_buffer = None
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

    @abstractmethod
    def learn(self, total_timesteps, log_interval):
        raise NotImplementedError

    def train(self):
        '''
        For deep models only; tabular models are trivial to implement.
        '''
        raise NotImplementedError

    def predict(self, observation, deterministic=False):
        observation = np.array(observation)

        action, value = self.policy.predict(observation, deterministic=deterministic)

        return action, value

    def save(self, save_path, **kwargs):
        mkdir_if_not_exist(save_path)
        self.policy.save(save_path)
        if self.replay_buffer is not None:
            self.replay_buffer.save(save_path)

        excluded = []
        excluded = self.excluded_params()

        to_save = self.__dict__.copy()
        for key in excluded:
            if key in to_save:
                del to_save[key]

        # print(to_save)
        # breakpoint()
        full_path = save_path + '/params/'
        mkdir_if_not_exist(full_path)
        with open(full_path + 'params.pkl', 'wb') as f:
            # print(to_save)
            pickle.dump(to_save, f)

    def load(self, load_path, env=None, **kwargs):
        self.policy.load(load_path)
        full_path = load_path + '/params/'
        mkdir_if_not_exist(full_path)
        with open(full_path + 'params.pkl', 'rb') as f:
            obj = pickle.load(f)
            for key, item in obj.items():
                if key in self.excluded_params():
                    continue
                try:
                    self.__dict__[key] = item
                except KeyError:
                    pass

    def excluded_params(self):
        return ["policy", "replay_buffer", "qvalues", "hvalues", "intention"]

    def get_qvalues(self):
        return self.qvalues
    
    def get_hvalues(self):
        return self.hvalues

class DeepRLModel(BaseRLModel):
    def __init__(
        self,
        policy,
         env,
        transform_func,
        gamma,
        learning_rate,
        buffer_size,
        exploration_type,
        exploration_frac,
        exploration_ep,
        exploration_initial_eps,
        exploration_final_eps, 
        double_q,
        policy_kwargs, 
        seed, 
        device
        ):

        super(DeepRLModel, self).__init__(
            policy=policy, env=env, 
            policy_kwargs=policy_kwargs,
            seed=seed
            )

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.exploration_type = exploration_type
        self.exploration_frac = exploration_frac
        self.exploration_ep = exploration_ep
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.double_q = double_q
        # self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        if device is None:
            self.device = get_device(device)
        else:
            self.device = device

        self.policy_kwargs = get_default_args(self.policy)
        self.policy_kwargs['ob_space'] = self.observation_space
        self.policy_kwargs['ac_space'] = self.action_space
        self.policy_kwargs['device'] = self.device
        self.policy_kwargs['learning_rate'] = self.learning_rate

        if policy_kwargs is not None:
            for key, val in policy_kwargs.items():
                self.policy_kwargs[key] = val
        # self.policy_kwargs['transform_func'] = transform_func

        # if policy_kwargs is None:
        #     self.policy = policy(self.observation_space, self.action_space,
        #                          intent=True, device=self.device)
        # else:
        self.policy = policy(**self.policy_kwargs)


        if self.buffer_size is None:
            self.replay_buffer = None
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size, device=self.device, torch=True)


    @abstractmethod
    def learn(self, total_timesteps, log_interval):
        raise NotImplementedError

    def train(self):
        '''
        For deep models only; tabular models are trivial to implement.
        '''
        raise NotImplementedError

    def predict(self, observation, deterministic=False):
        observation = np.array(observation)

        action, value = self.policy.predict(observation, deterministic=deterministic)

        return action, value

    def save(self, save_path, **kwargs):
        mkdir_if_not_exist(save_path)
        self.policy.save(save_path)
        if self.replay_buffer is not None:
            self.replay_buffer.save(save_path)

        excluded = []
        excluded = self.excluded_params()

        to_save = self.__dict__.copy()
        for key in excluded:
            if key in to_save:
                del to_save[key]

        # print(to_save)
        # breakpoint()
        full_path = save_path + '/params/'
        mkdir_if_not_exist(full_path)
        with open(full_path + 'params.pkl', 'wb') as f:
            # print(to_save)
            pickle.dump(to_save, f)

    def load(self, load_path, env=None, **kwargs):
        self.policy.load(load_path)
        full_path = load_path + '/params/'
        mkdir_if_not_exist(full_path)
        with open(full_path + 'params.pkl', 'rb') as f:
            obj = pickle.load(f)
            for key, item in obj.items():
                try:
                    self.__dict__[key] = item
                except KeyError:
                    pass

    def excluded_params(self):
        return ["policy", "device", "replay_buffer", "qvalues", "hvalues"]
