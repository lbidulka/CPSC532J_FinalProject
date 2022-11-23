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

from ribs.archives import ArchiveDataFrame
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

# Simulates the survivor in the environment and saves the trajectory 
def sim_survivor(env, survivor, seed:int=123):

    total_reward = 0.0
    state = env.reset(seed=seed)
    state = state[0]
    done = False

    step_count = 0
    step_limit = 50

    trajectory = []
    while not done:
        probs = survivor(torch.from_numpy(state))
        action = torch.argmax(probs)
        state, reward, done, info, _ = env.step(action.numpy())
        step_count += 1
        total_reward += reward
        trajectory.append(np.concatenate((state, action.numpy().reshape((-1,)), reward.reshape((-1,)))))
        if step_count >= step_limit:
            break
    trajectory = torch.from_numpy(np.stack(trajectory)).unsqueeze(0).float()

    return total_reward, trajectory, state, step_count

# Uses the system identifier network and the survivor trajectory to identify the env params
def identify(env, identifier, trajectory):
    # SysID 
    sys_pred = identifier(trajectory)
    pred_grav, pred_wp = (sys_pred[0,0].item(), sys_pred[0,1].item())

    return pred_grav, pred_wp

# Simulates the elite in the environment starting from where the survivor left off
def sim_elite(env, elite, state, step_count):
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    elite = elite.reshape((action_dim, obs_dim))
     
    total_reward = 0.0
    done = False

    while not done:
        action = np.argmax(elite @ state)  # Linear policy.
        state, reward, done, info, _ = env.step(action)
        step_count += 1
        total_reward += reward

        if step_count >= 950:
            break

    return total_reward

# Loads an elites policy from dataframe
def retrieve_elite(df, grav, wp, action_dim, obs_dim):
    df_dists = np.sqrt(np.power(df["behavior_0"]-grav,2) + np.power(df["behavior_1"]-wp,2)) 
    idx = df_dists.argsort()[:1]
    df_closest = df.iloc[idx] 
    elite_score = df_closest["objective"]
    elite = df_closest.loc[:, "solution_0":"solution_31"].to_numpy().reshape((action_dim, obs_dim))         # Extract model params

    return elite, elite_score

def main():
    seed = 123
    env = gym.make("LunarLander-v2")
    # Policy dims
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Load elite archive
    df = ArchiveDataFrame(pd.read_csv("./CPSC532J_FinalProject/lunar_lander_MAP_outputs/archive.csv"))

    # Test settings
    num_tests = 10
    USE_ORACLE = False  # The "Oracle" retrieves the exact elite required, without using the identifier network
    # Oracle Elite Region Parameters -----------------
    el_grav = -8.0    # default: -10.0, range: [-10, 0]
    el_wp = 5.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------
    # Env Parameters ---------------------------------
    grav = -8.0    # default: -10.0, range: [-10, 0]
    wp = 5.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------

    # Load survivor, identifier, and elite models
    survivor = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/Survivor-policy.pth")
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
        surv_reward, surv_trajectory, surv_end_state, surv_step_count = sim_survivor(env, survivor, seed)
        if USE_ORACLE:
            elite, elite_score = retrieve_elite(df, el_grav, el_wp, action_dim, obs_dim)
        else:
            pred_grav, pred_wp = identify(env, identifier, surv_trajectory)
            elite, elite_score = retrieve_elite(df, pred_grav, pred_wp, action_dim, obs_dim)
        elite_reward = sim_elite(env, elite, surv_end_state, surv_step_count)
        total_reward = surv_reward + elite_reward

        print(str(i) + ". Total Reward: ", total_reward, "Pred grav: ", pred_grav, "Pred wp: ", pred_wp)
        rewards.append(total_reward)
        grav_preds.append(pred_grav)
        wp_preds.append(pred_wp)

    print(" -------------------------------------\n")
    print("Avg Reward: ", sum(rewards)/len(rewards))
    print("Avg grav pred: ", sum(grav_preds)/len(grav_preds))
    print("Avg wp pred: ", sum(wp_preds)/len(wp_preds))

if __name__ == "__main__":
    main()