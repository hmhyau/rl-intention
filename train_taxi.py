from policies import TabularPolicy, DQNPolicy, IntentionPolicy, IntentionAblatedPolicy
from tabular_class import QTabularRLModel, MCTabularRLModel
from deep_class import DQNModel, IntentionModel, IntentionAblatedModel

import argparse
import numpy as np
import gym
from wrappers import DiscretizedObservationWrapper, TaxiObservationWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, help="Policy to train. One of [MC/Q/DQN].")
parser.add_argument("--load", type=str, help="Path to model to load.")
parser.add_argument("--ckpt", type=str, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, help="Random seed.")
parser.add_argument("--train", action="store_true", help="Training mode.")
args = parser.parse_args()

env = gym.make('Taxi-v3')

if args.policy in ["MC", "Q"]:
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
    env = TaxiObservationWrapper(env)
    model = IntentionAblatedModel(
        policy=IntentionAblatedPolicy,
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
