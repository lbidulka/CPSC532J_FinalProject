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

from models import Survivor, sysID, GA_Generalist
from sim_utils import sim_elite, sim_episode, sim_generalist_episode, sim_survivor, sim_tuning_episode, GA_tune, retrieve_elite

from ribs.archives import ArchiveDataFrame
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

# Zero shot application of elites to environment range
def EVAL_zero_shot(num_tests, env_range_res, seed, survivor, identifier, elite_archive_df):
    print("\n ---- 0-Shot Elite Appl. (", str(num_tests)," Runs / Setting)---- ")
    # Env setup
    grav_step, wp_step = env_range_res
    # env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)    # Create env of this type 
    grav_wp_rewards = []
    env_params = []
    grav_preds = []
    wp_preds = []
    for grav in tqdm(np.arange(-10.0, 0.0, grav_step)):
        wp_range_rewards = []
        for wp in np.arange(0, 20.0, wp_step):
            # print("Env Params | Gravity: ", grav, ", Wind Power: ", wp)
            rewards = []
            for i in range(num_tests):
                env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp) 
                total_reward, pred_grav, pred_wp = sim_episode(env, seed, survivor, identifier, elite_archive_df)
                grav_preds.append(pred_grav)
                wp_preds.append(pred_wp)    
                rewards.append(total_reward)
            wp_range_rewards.append(rewards)
        grav_wp_rewards.append(wp_range_rewards)

    print(" -----------------------------------------------")
    grav_wp_rewards_arr = np.array(grav_wp_rewards)
    return grav_wp_rewards_arr

# Random selection of elites after survivor agent
def EVAL_rand_ID(num_tests, env_range_res, seed, survivor, elite_archive_df):
    print("\n ---- Random Identifier (", str(num_tests)," Runs / Setting)---- ")
    # Env setup
    grav_step, wp_step = env_range_res
    grav_wp_rewards = []
    env_params = []
    grav_preds = []
    wp_preds = []
    for grav in tqdm(np.arange(-10.0, 0.0, grav_step)):
        wp_range_rewards = []
        for wp in np.arange(0, 20.0, wp_step):
            # print("Env Params | Gravity: ", grav, ", Wind Power: ", wp)
            rewards = []
            for i in range(num_tests):
                env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp) 
                total_reward, pred_grav, pred_wp = sim_episode(env, seed, survivor, "random", elite_archive_df)
                grav_preds.append(pred_grav)
                wp_preds.append(pred_wp)    
                rewards.append(total_reward) 
            wp_range_rewards.append(rewards)
        grav_wp_rewards.append(wp_range_rewards)

    print(" -----------------------------------------------")
    grav_wp_rewards_arr = np.array(grav_wp_rewards)
    return grav_wp_rewards_arr

# "Oracle" application of exact elites to environment range
def EVAL_oracle_ID(num_tests, env_range_res, seed, survivor, elite_archive_df):
    print("\n ---- Oracle Identifier (", str(num_tests)," Runs / Setting)---- ")
    # Env setup
    grav_step, wp_step = env_range_res
    grav_wp_rewards = []
    env_params = []
    grav_preds = []
    wp_preds = []
    for grav in tqdm(np.arange(-10.0, 0.0, grav_step)):
        wp_range_rewards = []
        for wp in np.arange(0, 20.0, wp_step):
            # print("Env Params | Gravity: ", grav, ", Wind Power: ", wp)
            rewards = []
            for i in range(num_tests):
                env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp) 
                total_reward, pred_grav, pred_wp = sim_episode(env, seed, survivor, "oracle", elite_archive_df)
                grav_preds.append(pred_grav)
                wp_preds.append(pred_wp)    
                rewards.append(total_reward) 
            wp_range_rewards.append(rewards)
        grav_wp_rewards.append(wp_range_rewards)

    print(" -----------------------------------------------")
    grav_wp_rewards_arr = np.array(grav_wp_rewards)
    return grav_wp_rewards_arr

# Generalist agent evaluation
def EVAL_Generalist(num_tests, env_range_res, seed, generalist):
    print("\n ---- Generalist (", str(num_tests)," Runs / Setting)---- ")
    # Env setup
    grav_step, wp_step = env_range_res
    grav_wp_rewards = []
    for grav in tqdm(np.arange(-10.0, 0.0, grav_step)):
        wp_range_rewards = []
        for wp in np.arange(0, 20.0, wp_step):
            rewards = []
            for i in range(num_tests):
                env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp) 
                total_reward = sim_generalist_episode(env, seed, generalist)
                rewards.append(total_reward) 
            wp_range_rewards.append(rewards)
        grav_wp_rewards.append(wp_range_rewards)

    print(" -----------------------------------------------")
    grav_wp_rewards_arr = np.array(grav_wp_rewards)
    return grav_wp_rewards_arr

# Fine tune application of elites to environment range
def EVAL_fine_tune(num_tests, env_range_res, seed, survivor, elite_archive_df):
    print("\n ---- Fine-Tune Elite Appl. (", str(num_tests)," Runs / Setting)---- ")
    # Env setup
    grav_step, wp_step = env_range_res
    # env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)    # Create env of this type 
    grav_wp_rewards = []
    env_params = []
    for grav in tqdm(np.arange(-10.0, 0.0, grav_step)):
        wp_range_rewards = []
        for wp in np.arange(0, 20.0, wp_step):
            env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp) 
            action_dim = env.action_space.n
            obs_dim = env.observation_space.shape[0]

            archive_elite, _ = retrieve_elite(elite_archive_df, grav, wp, action_dim, obs_dim)
            starting_elite = GA_Generalist()
            for param in starting_elite.parameters():
                param.data = torch.tensor(archive_elite, dtype=torch.float32)
            num_tuning_gens = 1 #5
            tuned_elite, max_rewards = GA_tune(env, starting_elite, num_tuning_gens, 25, 1, 30, 0.5, None)

            rewards = []
            for i in range(num_tests):
                total_reward = sim_tuning_episode(env, seed, survivor, tuned_elite)
                rewards.append(total_reward)
            wp_range_rewards.append(rewards)
        grav_wp_rewards.append(wp_range_rewards)

    print(" -----------------------------------------------")
    grav_wp_rewards_arr = np.array(grav_wp_rewards)
    return grav_wp_rewards_arr

def main():
    seed = 123
    log_dir = "./CPSC532J_FinalProject/src/logs/"

    # Load elite archive
    elite_archive_df = ArchiveDataFrame(pd.read_csv("./CPSC532J_FinalProject/lunar_lander_MAP_outputs/archive.csv"))

    # Test settings
    num_tests = 1000
    num_envparam_steps = 7  # 7 minimum
    env_range_res = (10/num_envparam_steps, 20/num_envparam_steps)
    np.save(log_dir + "experiments/env_range_res.npy", np.array(env_range_res))

    # Load survivor, identifier, and GA-generalist models
    survivor = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/Survivor-prerefactor.pth")
    survivor.eval()
    identifier = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/sysID.pth").to("cpu")
    identifier.eval()    
    GA_generalist = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/GA-general-policy.pth")
    GA_generalist.eval()

    # # Run some experiments
    # zero_shot_rewards = EVAL_zero_shot(num_tests, env_range_res, seed, survivor, identifier, elite_archive_df)
    # np.save(log_dir + "experiments/zero_shot_rewards.npy", zero_shot_rewards)

    # rand_ID_rewards = EVAL_rand_ID(num_tests, env_range_res, seed, survivor, elite_archive_df)
    # np.save(log_dir + "experiments/rand_ID_rewards.npy", rand_ID_rewards)

    # oracle_ID_rewards = EVAL_oracle_ID(num_tests, env_range_res, seed, survivor, elite_archive_df)
    # np.save(log_dir + "experiments/oracle_ID_rewards.npy", oracle_ID_rewards)

    # GA_generalist_rewards = EVAL_Generalist(num_tests, env_range_res, seed, GA_generalist)
    # np.save(log_dir + "experiments/GA_generalist_rewards.npy", GA_generalist_rewards)

    tuned_rewards = EVAL_fine_tune(num_tests, env_range_res, seed, survivor, elite_archive_df)
    np.save(log_dir + "experiments/tuned_rewards.npy", tuned_rewards)


if __name__ == "__main__":
    main()