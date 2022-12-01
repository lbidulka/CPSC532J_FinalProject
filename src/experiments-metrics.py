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

def plot_heatmap(value, log_dir, out_name, title, cmap_bounds, cmap='viridis'):
    # format is [grav variation, wp variation, value for each run]
    # med_vals = np.median(value, axis=2)
    reshaped_med_vals = np.swapaxes(value, 0, 1)
    reshaped_med_vals = np.flip(reshaped_med_vals, axis=0)

    plt.imshow(reshaped_med_vals, 
                interpolation="none",
                extent=[-10, 0, 0, 20], 
                vmin=cmap_bounds[0], 
                vmax=cmap_bounds[1], 
                cmap=cmap)
    plt.colorbar()
    plt.ylabel("wp")
    plt.xlabel("grav")
    plt.title(title + " Heatmap")
    plt.savefig(log_dir + "plots/" + out_name + "_heatmap.jpg")
    # plt.show()
    plt.clf()

def main():
    log_dir = "./CPSC532J_FinalProject/src/logs/experiments/"
    rewards_files = ["zero_shot_rewards.npy", "rand_ID_rewards.npy", "oracle_ID_rewards.npy", "GA_generalist_rewards.npy", "tuned_rewards.npy"]
    titles = ["0-Shot Mnemosyne Solve Rate", "Random ID Solve Rate", "Oracle ID Solve Rate", "GA Generalist Solve Rate", "Fine-tuned Mnemosyne Solve Rate"]
    out_names = ["zero_shot_solve", "rand_id_solve", "oracle_id_solve", "ga_generalist_solve", "tuned_solve"]
    # env_range_res = np.load(log_dir + "env_range_res.npy")

    # Plot heatmaps and some metrics
    for rewards_file, title, out_name in zip(rewards_files, titles, out_names):
        rewards = np.load(log_dir + rewards_file)
        solve_rates = np.sum(rewards >= 200, axis=2) / rewards.shape[2]
        plot_heatmap(solve_rates, log_dir, out_name, title, (0, 1), cmap='cividis')    # env cell solution rate plotting
        print("\n----- " + out_name[:-6] + " -----")
        print(" overall env solve rate", round(np.sum(rewards >= 200) / rewards.size, 3))
        print(" reward median: ", np.median(np.reshape(rewards, (-1,))))

if __name__ == "__main__":
    main()