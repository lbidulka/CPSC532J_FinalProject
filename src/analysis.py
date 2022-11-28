import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from models import Survivor, sysID
from sim_utils import sim_elite, sim_episode, sim_generalist_episode, sim_survivor

from ribs.archives import ArchiveDataFrame
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

def main():
    seed = 123
    env = gym.make("LunarLander-v2")
    # Policy dims
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Load elite archive
    elite_archive_df = ArchiveDataFrame(pd.read_csv("./CPSC532J_FinalProject/lunar_lander_MAP_outputs/archive.csv"))

    # Test settings
    num_tests = 3
    USE_ORACLE = False  # The "Oracle" retrieves the exact elite required, without using the identifier network
    GENERALIST = True
    # Env Parameters ---------------------------------
    grav = -10.0    # default: -10.0, range: [-10, 0]
    wp = 20.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------
    # Oracle Elite Region Parameters -----------------
    el_grav = -8.0    # default: -10.0, range: [-10, 0]
    el_wp = 5.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------

    # Load survivor, identifier, and elite models
    print("-- GENERALIST --")   
    print(GENERALIST)
    if GENERALIST:
        GA_generalist = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/GA-general-policy.pth")
        GA_generalist.eval()
    if not GENERALIST:
        survivor = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/Survivor.pth") # survivor options = ["random", "oracle", policy network]
        survivor.eval()
        identifier = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/sysID.pth").to("cpu")
        identifier.eval()    

    print("-- ORACLE --")   
    print(USE_ORACLE)
    if USE_ORACLE:
        print("-- Oracle Elite Params --")
        print("Gravity: ", el_grav, " Wind Power: ,", el_wp)

    # Create env and run some simulations to evaluate
    print("-- Env Params --")
    print("Gravity: ", grav, " Wind Power: ,", wp)
    env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)    # Create env of this type
    rewards = []
    grav_preds = []
    wp_preds = []
    print("\n ---- Simulation Results (", str(num_tests)," Runs)---- ")
    for i in range(num_tests):
        if GENERALIST:
            total_reward = sim_generalist_episode(env, seed, GA_generalist)
            print(str(i) + ". Total Reward: ", total_reward)
        if not GENERALIST:
            total_reward, pred_grav, pred_wp = sim_episode(env, seed, survivor, identifier, elite_archive_df)
            grav_preds.append(pred_grav)
            wp_preds.append(pred_wp)
            print(str(i) + ". Total Reward: ", total_reward, "Pred grav: ", pred_grav, "Pred wp: ", pred_wp)
        rewards.append(total_reward)

    print(" -------------------------------------\n")
    print("Avg Reward: ", sum(rewards)/len(rewards))
    if not GENERALIST:
        print("Avg grav pred: ", sum(grav_preds)/len(grav_preds))
        print("Avg wp pred: ", sum(wp_preds)/len(wp_preds))

if __name__ == "__main__":
    main()