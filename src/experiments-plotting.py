import gym
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
import argparse
import copy

import torch
from torch import nn
import torch.nn.functional as F

def plot_reward_heatmap(zero_shot_rewards, env_range_res, log_dir):
    grav_step, wp_step = env_range_res
    # zero_shot_rewards is [grav variation, wp variation, rewards for each run]
    mean_rewards = np.mean(zero_shot_rewards, axis=2)
    reshaped_mean_rewards = np.swapaxes(mean_rewards, 0, 1)
    reshaped_mean_rewards = np.flip(reshaped_mean_rewards, axis=0)

    # plt.imshow(mean_rewards, interpolation="none", extent=[0, -10, 0, 20,], aspect=1)
    plt.imshow(reshaped_mean_rewards, interpolation="none",extent=[-10, 0, 0, 20])
    plt.colorbar()
    plt.ylabel("wp")
    plt.xlabel("grav")
    
    plt.title("0-Shot Mnemosyne Reward Heatmap")
    plt.savefig(log_dir + "plots/reward_heatmap.jpg")
    plt.show()


def main():

    # Load results
    log_dir = "./CPSC532J_FinalProject/src/logs/experiments/"
    zero_shot_rewards = np.load(log_dir + "zero_shot_rewards.npy")
    env_range_res = np.load(log_dir + "env_range_res.npy")

    # Plot
    plot_reward_heatmap(zero_shot_rewards, env_range_res, log_dir)

if __name__ == "__main__":
    main()