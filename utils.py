import matplotlib.pyplot as plt
import seaborn as sns

import random
import numpy as np
import gym
import torch
import json
import sys
import inspect

import os
import os.path as osp
from functools import reduce

def plot_belief(data, title, figsize, ):
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(data, linewidth=1, center=center, cmap=cmap, xticklabels=20, yticklabels=['left', 'right'])
    ax.set_title(title)
    ax.set_ylim(2, 0)
    ax.set_ylabel('Actions')
    ax.set_xlabel('States')
    # ax.set_xticks(np.arange(70, 91, 1))
    ax.tick_params(labelsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # f.tight_layout()
    f.subplots_adjust(top=0.8, bottom=0.3, right=0.8)
    f.tight_layout()
    return f

def plot_table_blackjack(data, title, center=None, figsize=(7.5, 12), cmap=None):
    '''
    Flatten from 4-D to 2-D and plot all heatmaps.
    '''
    TITLE = ['Stick, No Usable Ace', 'Stick, With Usable Ace', 'Hit, No Usable Ace', 'Hit, With Usable Ace']
    # if contrast:
    #     cmap = sns.diverging_palette(10, 240, n=128)
    #     center = 0
    # else:
    #     cmap = 'Blues'
    cmap = 'Blues' if cmap is None else cmap

    # f, ax = plt.subplots(figsize=figsize)
    nrows = 2
    ncols = 2
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), constrained_layout=True)
    
    to_plot = np.split(data, data.shape[-1], axis=-1)
    to_plot = [np.squeeze(d) for d in to_plot]
    
    # breakpoint()
    to_plot = [np.split(d, d.shape[-1], axis=-1) for d in to_plot]
    to_plot = [np.squeeze(t) for sub in to_plot for t in sub]
    # print(to_plot[0].shape)
    for idx, (ax, plot) in enumerate(zip(axes.flatten(), to_plot)):
        # print(plot)
        # ax = sns.heatmap(plot, center=center, linewidth=1, yticklabels=1, cmap=cmap)
        sns.heatmap(plot, center=center, linewidth=1, yticklabels=1, cmap=cmap, ax=ax, cbar_kws={"fraction": 0.1, "pad": 0.1, "aspect": 40})
        ax.set_title(TITLE[idx])
        # States outside this range are unreachable
        ax.set_ylim(22, 4)
        ax.set_xlim(1, 11)
        ax.set_ylabel('Sum of Player Hand')
        ax.set_xlabel('Dealer Face-up Card')
        ax.tick_params(labelsize=10)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
    return f

def plot_outcome_blackjack(data, title, figsize=(10,5.4), center=None, cmap=None):
    # if contrast:
    #     cmap = sns.diverging_palette(10, 240, n=128)
    #     center = 0
    # else:
    #     cmap = 'Blues'
    
    cmap = 'Blues' if cmap is None else cmap

    data = np.expand_dims(data[..., :-2], axis=0)
    f, ax = plt.subplots(figsize=figsize)
#     xticklabels = ['Stick & lose', 'Stick & win', 'Hit & lose', 'Stick & draw', 'Hit only']
    xticklabels = ['Stick & lose', 'Stick & win', 'Hit & lose', 'Stick & draw']
    ax = sns.heatmap(data, center=center, linewidth=3, cmap=cmap, xticklabels=xticklabels, cbar_kws={"orientation": 'horizontal', "pad": 0.35}, yticklabels=False, annot_kws={"size":16})
    ax.set_title(title)
    ax.set_ylabel('Belief')
    ax.set_xlabel('Outcomes')
    ax.tick_params(labelsize=20)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    return f

def plot_table_cartpole(data, title, figsize, contrast=False):
    if contrast:
        cmap = sns.diverging_palette(10, 240, n=128)
        center = 0
    else:
        cmap = 'Blues'

    f, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(data, linewidth=1, center=center, cmap=cmap, xticklabels=20, yticklabels=['left', 'right'])
    ax.set_title(title)
    ax.set_ylim(2, 0)
    ax.set_ylabel('Actions')
    ax.set_xlabel('States')
    # ax.set_xticks(np.arange(70, 91, 1))
    ax.tick_params(labelsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # f.tight_layout()
    f.subplots_adjust(top=0.8, bottom=0.3, right=0.8)
    f.tight_layout()
    return f
    

def plot_belief_taxi(data):
    pass

def plot_contrast_taxi(belief, con_belief):
    pass

def set_global_seeds(seed):
    # set numpy and random seeds
    np.random.seed(seed)
    random.seed(seed)

    # Set gym env seed
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)

    # Set torch seed
    try:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed) 
    except ImportError:
        pass

def get_device(device = None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    return device

def convert_to_onehot(array, size, transform):
    # print(array.long().size())
    out = torch.zeros(array.size()[0]).cuda()
    if isinstance(array, torch.Tensor):
        for idx, arr in enumerate(array):
            int_ = transform([3, 3, 6, 3], arr)
            # breakpoint()
            if isinstance(int_, np.ndarray):
                int_ = torch.as_tensor(int_)
            # breakpoint()
            out[idx] = int_

        out = out.reshape(-1, 1)
        onehot = torch.FloatTensor(array.size()[0], 162).cuda()
        onehot.zero_()
        onehot.scatter_(1, out.long(), 1)
        return onehot

def is_arraylike(x):
    if isinstance(x, (list, np.ndarray)):
        return True
    return False

def as_list(x):
    if is_arraylike(x):
        return x
    else:
        return np.array([x])

def identity(x):
    return x

def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v, in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def mkdir_if_not_exist(path):
    if not osp.exists(path):
        os.makedirs(path)

class CartpoleEncoder:
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def __call__(self, digits):
        out = 0
        for i in reversed(range(len(digits))):
            if i == 0:
                out += digits[-(i+1)]
            else:
                out += reduce(lambda x, y: x*y, self.n_bins[-i:]) * digits[-(i+1)]
        return out

def encode_cartpole(n_bins, digits):
    out = 0
    for i in reversed(range(len(digits))):
        if i == 0:
            out += digits[-(i+1)]
        else:
            out += reduce(lambda x, y: x*y, n_bins[-i:]) * digits[-(i+1)]
    return out

# def save_to_zip(save_path, data=None, params=None, torch_model=False, np_model=False):
#     assert torch_model != np_model
#     assert os.path.exists(save_path)

#     with zipfile.ZipFile(save_path, mode="w") as archive:
#         if params is not None:
#             with archive.open("")