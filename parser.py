import argparse

def make_cmd_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help="learning rate of the model")
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--buffer", type=int, help="replay buffer, >1 = use buffer")
    parser.add_argument("--exploration", type=str, help="exploration schedule. One of [linear/exp]")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--policy", type=str, help="policy to use. One of [MC/Q]")
    parser.add_argument("--ckpt", type=str, help="path to ckpt")
    parser.add_argument("--intent", action='store_true', help="enable intention training")

    return parser