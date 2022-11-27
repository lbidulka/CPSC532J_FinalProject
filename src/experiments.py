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

# Simulates a full survivor -> identifier -> elite episode
def sim_episode(env, seed, survivor, identifier, elite_archive_df):
    surv_reward, surv_trajectory, surv_end_state, surv_step_count = sim_survivor(env, survivor, seed)
    pred_grav, pred_wp = identify(env, identifier, surv_trajectory)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    elite, elite_score = retrieve_elite(elite_archive_df, pred_grav, pred_wp, action_dim, obs_dim)
    elite_reward = sim_elite(env, elite, surv_end_state, surv_step_count)
    total_reward = surv_reward + elite_reward
    return total_reward, pred_grav, pred_wp

# Loads an elites policy from dataframe
def retrieve_elite(df, grav, wp, action_dim, obs_dim):
    df_dists = np.sqrt(np.power(df["behavior_0"]-grav,2) + np.power(df["behavior_1"]-wp,2)) 
    idx = df_dists.argsort()[:1]
    df_closest = df.iloc[idx] 
    elite_score = df_closest["objective"]
    elite = df_closest.loc[:, "solution_0":"solution_31"].to_numpy().reshape((action_dim, obs_dim))         # Extract model params

    return elite, elite_score

# Zero shot application of elites to environment range
def EVAL_zero_shot(num_tests, env_range_res, seed, survivor, identifier, elite_archive_df):
    print("\n ---- 0-Shot Elite Appl. (", str(num_tests)," Runs)---- ")
    # Env setup
    grav_step, wp_step = env_range_res
    # env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)    # Create env of this type 
    grav_wp_rewards = []
    env_params = []
    grav_preds = []
    wp_preds = []
    for grav in np.arange(-10.0, 0.0, grav_step):
        wp_range_rewards = []
        for wp in np.arange(0, 20.0, wp_step):
            print("Env Params | Gravity: ", grav, ", Wind Power: ", wp)
            rewards = []
            for i in range(num_tests):
                env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp) 
                total_reward, pred_grav, pred_wp = sim_episode(env, seed, survivor, identifier, elite_archive_df)
                grav_preds.append(pred_grav)
                wp_preds.append(pred_wp)    
                rewards.append(total_reward)
            # avg_reward = sum(rewards)/len(rewards)
            # print("Avg Reward: ", avg_reward)   
            wp_range_rewards.append(rewards)
        grav_wp_rewards.append(wp_range_rewards)

    print(" ---------------------------------------\n")
    grav_wp_rewards_arr = np.array(grav_wp_rewards)
    print(grav_wp_rewards_arr)
    print(grav_wp_rewards_arr.shape)
    return grav_wp_rewards_arr

def main():
    seed = 123
    log_dir = "./CPSC532J_FinalProject/src/logs/"

    # Load elite archive
    elite_archive_df = ArchiveDataFrame(pd.read_csv("./CPSC532J_FinalProject/lunar_lander_MAP_outputs/archive.csv"))

    # Test settings
    num_tests = 100
    num_envparam_steps = 7  # 7 minimum
    env_range_res = (10/num_envparam_steps, 20/num_envparam_steps)

    # Load survivor, identifier, and elite models
    survivor = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/Survivor-prerefactor.pth")
    survivor.eval()
    identifier = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/sysID.pth").to("cpu")
    identifier.eval()    

    # Run some experiments
    zero_shot_rewards = EVAL_zero_shot(num_tests, env_range_res, seed, survivor, identifier, elite_archive_df)
    np.save(log_dir + "experiments/zero_shot_rewards.npy", zero_shot_rewards)
    np.save(log_dir + "experiments/env_range_res.npy", np.array(env_range_res))


if __name__ == "__main__":
    main()