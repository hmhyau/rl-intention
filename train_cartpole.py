from policies import TabularPolicy, DQNPolicy, IntentionPolicy
from tabular_class import QTabularRLModel
from deep_class import DQNModel, IntentionModel

import argparse
import numpy as np
import gym
from wrappers import DiscretizedObservationWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, help="Policy to train. One of [MC/Q/DQN].")
parser.add_argument("--load", type=str, help="Path to model to load.")
parser.add_argument("--ckpt", type=str, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, help="Random seed.")
parser.add_argument("--train", action="store_true", help="Training mode.")
args = parser.parse_args()

env = gym.make('CartPole-v0')

if args.policy in ["MC", "Q"]:
    env = DiscretizedObservationWrapper(
        env,
        n_bins=np.array([3, 3, 6, 3]),
        low=[-2.4, -2.5, -np.radians(12), -1],
        high=[2.4, 2.5, np.radians(12), 1],
        convert=True
        )
    if args.policy == 'Q':
        model = QTabularRLModel(
            policy=TabularPolicy,
            env=env,
            learning_rate=0.1,
            gamma=1.,
            exploration_type="linear",
            exploration_frac=0.999,
            exploration_initial_eps=1.,
            exploration_final_eps=0.05,
            seed=args.seed,
            intent=True)

    if args.policy == 'MC':
        model = MCTabularRLModel(
            policy=TabularPolicy,
            env=env,
            learning_rate=0.1,
            gamma=1.,
            exploration_type="linear",
            exploration_frac=0.999,
            exploration_initial_eps=1.,
            exploration_final_eps=0.05,
            seed=args.seed,
            intent=True)

if args.policy == 'DQN':
    env = DiscretizedObservationWrapper(
        env,
        n_bins=np.array([3, 3, 6, 3]),
        low=[-2.4, -2.5, -np.radians(12), -1],
        high=[2.4, 2.5, np.radians(12), 1],
        convert=False
        )
    model = IntentionModel(
        policy=IntentionPolicy,
        env=env,
        learning_rate=0.0001,
        gamma=1.,
        buffer_size=10000,
        exploration_type="linear",
        exploration_frac=0.999,
        exploration_initial_eps=1.,
        exploration_final_eps=0.05,
        seed=args.seed)

    model.set_random_seed(args.seed)
    if args.load:
        model.load(args.load)
    if args.train:
        model.learn(total_episodes=5000, ckpt_interval=100, ckpt_path=args.ckpt)

# model.load(load_path='./cartpole/dqn/3000/')
