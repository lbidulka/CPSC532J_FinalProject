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
    med_vals = np.median(value, axis=2)
    reshaped_med_vals = np.swapaxes(med_vals, 0, 1)
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

def plot_hist(values, log_dir, out_name, title, num_bins=80):
    flat_vals = np.reshape(values, (-1,))
    plt.hist(flat_vals, bins=num_bins)
    plt.title(title + " Histogram")
    plt.ylabel("Num occurances")
    plt.xlabel("Reward")
    plt.savefig(log_dir + "plots/" + out_name + "_histogram.jpg")
    # plt.show()
    plt.clf()

def main():
    log_dir = "./CPSC532J_FinalProject/src/logs/experiments/"
    rewards_files = ["zero_shot_rewards.npy", "rand_ID_rewards.npy", "oracle_ID_rewards.npy", "GA_generalist_rewards.npy", "tuned_rewards.npy"]
    titles = ["0-Shot Mnemosyne Reward", "Random ID Reward", "Oracle ID Reward", "GA Generalist Reward", "Fine-tuned Mnemosyne Reward"]
    out_names = ["zero_shot_reward", "rand_id_reward", "oracle_id_reward", "ga_generalist_reward", "tuned_reward"]
    # env_range_res = np.load(log_dir + "env_range_res.npy")

    # Plot heatmaps and histograms
    for rewards_file, title, out_name in zip(rewards_files, titles, out_names):
        rewards = np.load(log_dir + rewards_file)
        plot_heatmap(rewards, log_dir, out_name, title, (-100, 400))
        plot_hist(rewards, log_dir, out_name, title)
        print(out_name + " reward median: ", np.median(np.reshape(rewards, (-1,))))

    # Plot sysID
    grav_wp_preds = np.load(log_dir + "sysID_preds.npy")
    grav_wp_gt = np.load(log_dir + "sysID_gt.npy")
    grav_wp_errs = np.abs(grav_wp_preds - grav_wp_gt)
    grav_medians = np.median(grav_wp_errs[:,:,0], axis=2)
    wp_medians = np.median(grav_wp_errs[:,:,1], axis=2)
    plot_heatmap(grav_wp_errs[:,:,0], 
                    log_dir, 
                    "sysID_grav_err", 
                    "SysID Grav Error Heatmap", 
                    (0, np.max(grav_medians)), 
                    cmap="plasma")
    plot_heatmap(grav_wp_errs[:,:,1], 
                    log_dir, 
                    "sysID_wp_err", 
                    "SysID wp Error Heatmap", 
                    (0, np.max(wp_medians)),
                    cmap="plasma")


if __name__ == "__main__":
    main()