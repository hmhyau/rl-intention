from functools import reduce

import numpy as np

import gym
from gym import ObservationWrapper, ActionWrapper, Wrapper
from gym.spaces import Box, Discrete, Tuple
from copy import deepcopy

class DiscretizedObservationWrapper(ObservationWrapper):
    def __init__(self, env, n_bins=None, low=None, high=None, convert=True):
        super().__init__(env)
        self.convert = convert
        assert isinstance(env.observation_space, Box)
        self.obs_shape = self.observation_space.shape
        assert n_bins.shape == self.obs_shape

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.disc_bins = [np.linspace(l, h, bin+1) for l, h, bin in 
                          zip(low.flatten(), high.flatten(), n_bins)]
        self.disc_r = [np.linspace(l, h, bin+1)[1:-1] for l, h, bin in
                       zip(low.flatten(), high.flatten(), n_bins)]

        # preserve original observation space info
        self.orig_observation_space = deepcopy(self.observation_space)
        if convert:
            self.observation_space = Discrete(np.prod(self.n_bins))
        else:
            self.observation_space = Tuple([Discrete(x) for x in self.n_bins])

    def _convert_to_int(self, digits):
        out = 0
        for i in reversed(range(len(self.disc_bins))):
            if i == 0:
                out += digits[-(i+1)]
            else:
                out += reduce(lambda x, y: x*y, self.n_bins[-i:]) * digits[-(i+1)]

        return out

    def observation(self, observation):
        digits = [np.digitize(x=x, bins=bins) for x, bins in zip(observation.flatten(), self.disc_r)]
        # print(digits, self.disc_r)
        if self.convert:
            return self._convert_to_int(digits)
        
        return np.array(digits).astype(np.float32)

class BlackjackTorchObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _convert_bool_to_float(self, state):
        return tuple(map(float, state))
    
    def observation(self, observation):
        return np.array(self._convert_bool_to_float(observation))

class BlackjackTabularObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _convert_bool_to_int(self, state):
        return tuple(map(int, state))
    
    def observation(self, observation):
        return self._convert_bool_to_int(observation)

class TaxiObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Tuple((Discrete(5), Discrete(5), Discrete(5), Discrete(4)))
    
    def observation(self, observation):
        return np.array(tuple(map(float, list(self.env.decode(observation))))).astype(np.float32)

class TransformObservation(ObservationWrapper):
    r"""Transform the observation via an arbitrary function. 
    Example::
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformObservation(env, lambda obs: obs + 0.1*np.random.randn(*obs.shape))
        >>> env.reset()
        array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
    Args:
        env (Env): environment
        f (callable): a function that transforms the observation
    """
    def __init__(self, env, f):
        super(TransformObservation, self).__init__(env)
        assert callable(f)
        self.f = f

    def observation(self, observation):
        return self.f(observation)

# class CusomDiscretizedActWrapper(ActionWrapper):
#     def __init__(self):
#         pass

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    env = DiscretizedObservationWrapper(env,
                                    n_bins=np.array([3, 3, 6, 3]),
                                    low=[-2.4, -2.0, -0.4, -3.5],
                                    high=[2.4, 2.0, 0.4, 3.5], convert=True)
    # env = gym.make('MountainCar-v0')
    # env = DiscretizedObservationWrapper(env,
    #                             n_bins=np.array([4, 4]),
    #                             low=[-1.2, 0.6],
    #                             high=[-0.07, 0.07])
    breakpoint()
    print(env.reset())
    # print(env.step())
    
    print(env.unwrapped.state)