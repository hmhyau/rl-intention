from blackjack_tabular_class import BlackjackQTabularRLModel, BlackjackMCTabularRLModel
from policies import BlackjackTabularPolicy
from wrappers import BlackjackTabularObservationWrapper
from utils import get_default_args

import argparse
import numpy as np
import gym

def compute_q_from_intention(intention):
    recovery = np.zeros(model.qvalues.shape)
    r = np.array([-1, 1, -1, 0, 0])
    for idx, x in np.ndenumerate(recovery):
        state = tuple(idx[0:-1])
        val = np.sum(model.intention[idx] * r)
        recovery[idx] = val
    
    # Cannot check consistency with exact equivlency due to precision
    assert np.all(np.isclose(recovery, model.qvalues)), "Consistency Violated"
    
    return recovery

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, help="Policy to train. One of [MC/Q].")
    parser.add_argument("--load", type=str, help="Path to model to load.")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--train", action="store_true", help="Training mode.")
    args = parser.parse_args()

    env = BlackjackTabularObservationWrapper(gym.make('Blackjack-v0'))

    if args.policy == "MC":
        # Tabular MC Control
        model = BlackjackMCTabularRLModel(
            policy=BlackjackTabularPolicy,
            env=env,
            learning_rate=0.1,
            gamma=1.,
            exploration_type="exponential",
            exploration_frac=0.9999,
            exploration_initial_eps=1.,
            exploration_final_eps=0.05,
            seed=args.seed,
            intent=True)
        
    if args.policy == "Q":
        # Tabular Q-learning
        model = BlackjackQTabularRLModel(
            policy=BlackjackTabularPolicy,
            env=env,
            learning_rate=.1,
            gamma=1.,
            exploration_type="exponential",
            exploration_frac=0.9999,
            exploration_initial_eps=1.,
            exploration_final_eps=0.05,
            seed=args.seed,
            intent=True)
        model.set_random_seed(args.seed)

    model.set_random_seed(args.seed)
    if args.load:
        model.load(load_path=args.load)
    if args.train:
        model.learn(
            total_episodes=500000,
            ckpt_interval=100000,
            ckpt_path=args.ckpt)


    # Some sanity check
    assert(id(model.policy.hvalues) == id(model.hvalues))

    from utils import plot_table_blackjack, plot_outcome_blackjack
    import seaborn as sns
    import matplotlib.pyplot as plt

    recovered = compute_q_from_intention(model.intention)
    table_cmap = sns.diverging_palette(10, 240, n=128)
    plot_table_blackjack(model.qvalues, center=0, title="temp", cmap=table_cmap)
    # plot_table_blackjack(recovered[..., 0, 0], center=0, title='recovered', cmap=table_cmap)
    # plot_table_blackjack(model.qvalues[..., 0, 0], center=0, title='original', cmap=table_cmap)
    # plot_table_blackjack(recovered[..., 1, 0], center=0, title='recovered', cmap=table_cmap)
    # plot_table_blackjack(model.qvalues[..., 1, 0], center=0, title='original', cmap=table_cmap)
    # plot_table_blackjack(recovered[..., 0, 1], center=0, title='recovered', cmap=table_cmap)
    # plot_table_blackjack(model.qvalues[..., 0, 1], center=0, title='original', cmap=table_cmap)
    # plot_table_blackjack(recovered[..., 1, 1], center=0, title='recovered', cmap=table_cmap)
    # plot_table_blackjack(model.qvalues[..., 1, 1], center=0, title='original', cmap=table_cmap)

    # print(model.hvalues[12, 6, 0, 1].shape)
    # print(model.hvalues[12, 6, 0, 1].sum(axis=-1).sum(axis=-1).max())
    # plot_table_blackjack(model.hvalues[10, 7, 0, 1].sum(axis=-1).sum(axis=-1), "Belief")
    # plot_table_blackjack(model.hvalues[10, 7, 0, 1][..., 0, 1], "Belief")
    # plot_table_blackjack(model.hvalues[10, 7, 0, 1][..., 0, 0], "Belief")
    # print(model.hvalues[10, 7, 0, 0].sum(axis=-1).sum(axis=-1).max())
    # plot_table_blackjack(model.hvalues[10, 7, 0, 0].sum(axis=-1).sum(axis=-1), "Belief")
    # plot_table_blackjack(model.hvalues[10, 7, 0, 0][..., 0, 1], "Belief")
    # plot_table_blackjack(model.hvalues[10, 7, 0, 0][..., 0, 0], "Belief")
    # print(model.intention[((17, 0, 0), 1)].shape)
    # print(model.intention.keys())
    # plot_outcome_blackjack(model.intention[((10, 7, 0), 0)], center=None, title="Outcome Belief")
    # plot_outcome_blackjack(model.intention[((10, 7, 0), 1)], center=None, title="Outcome Belief")

    # print(model.intention[((10, 7, 0), 1)].sum(), model.hvalues[10, 7, 0, 1].sum())
    plt.show()