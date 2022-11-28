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

def plot_reward_heatmap(zero_shot_rewards, env_range_res, log_dir, out_name, title):
    grav_step, wp_step = env_range_res
    # zero_shot_rewards is [grav variation, wp variation, rewards for each run]
    mean_rewards = np.mean(zero_shot_rewards, axis=2)
    reshaped_mean_rewards = np.swapaxes(mean_rewards, 0, 1)
    reshaped_mean_rewards = np.flip(reshaped_mean_rewards, axis=0)

    plt.imshow(reshaped_mean_rewards, interpolation="none",extent=[-10, 0, 0, 20], vmin=-400, vmax=400)
    plt.colorbar()
    plt.ylabel("wp")
    plt.xlabel("grav")
    
    plt.title(title)
    plt.savefig(log_dir + "plots/" + out_name + "_reward_heatmap.jpg")
    # plt.show()
    plt.clf()


def main():
    log_dir = "./CPSC532J_FinalProject/src/logs/experiments/"
    rewards_files = ["zero_shot_rewards.npy", "rand_ID_rewards.npy", "oracle_ID_rewards.npy", "GA_generalist_rewards.npy"]
    heatmap_titles = ["0-Shot Mnemosyne Reward Heatmap", "Random ID Reward Heatmap", "Oracle ID Reward Heatmap", "GA Generalist Reward Heatmap"]
    out_names = ["zero_shot", "rand_id", "oracle_id", "ga_generalist"]
    env_range_res = np.load(log_dir + "env_range_res.npy")

    # Plot
    for rewards_file, title, out_name in zip(rewards_files, heatmap_titles, out_names):
        rewards = np.load(log_dir + rewards_file)
        plot_reward_heatmap(rewards, env_range_res, log_dir, out_name, title=title)
        print(out_name + " reward median: ", np.median(np.reshape(rewards, (-1,))))

if __name__ == "__main__":
    main()