import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd

from ribs.archives import ArchiveDataFrame
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap


def simulate(env, model, seed:int=123):
    """Simulates the lunar lander model.

    Args:
        env (gym.Env): A copy of the lunar lander environment.
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
    """

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    obs = env.reset(seed=seed)
    obs = obs[0]
    done = False

    step_count = 0

    while not done:
        action = np.argmax(model @ obs)  # Linear policy.
        obs, reward, done, info, _ = env.step(action)
        step_count += 1
        total_reward += reward

        if step_count >= 1000:
            break

    return total_reward

def retrieve_elite(df, grav, wp, action_dim, obs_dim):
    df_dists = np.sqrt(np.power(df["behavior_0"]-grav,2) + np.power(df["behavior_1"]-wp,2)) 
    idx = df_dists.argsort()[:1]
    df_closest = df.iloc[idx] 
    elite_score = df_closest["objective"]
    elite = df_closest.loc[:, "solution_0":"solution_31"].to_numpy().reshape((action_dim, obs_dim))         # Extract model params

    return elite, elite_score


def main():
    # Init env with parameters
    seed = 123
    env = gym.make("LunarLander-v2")
    # Policy dims
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Load elite archive
    df = ArchiveDataFrame(pd.read_csv("./CPSC532J_FinalProject/lunar_lander_MAP_outputs/archive.csv"))

    # Run some demos
    num_tests = 10

    # Elite Region Parameters ------------------------
    el_grav = -10.0    # default: -10.0, range: [-10, 0]
    el_wp = 0.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------
    # Env Parameters ---------------------------------
    grav = -10.0    # default: -10.0, range: [-10, 0]
    wp = 0.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------

    elite, elite_score = retrieve_elite(df, el_grav, el_wp, action_dim, obs_dim)

    print("-- Elite Params --")
    print("Gravity: ", el_grav, " Wind Power: ,", el_wp)
    print("Recorded score: ", elite_score) 

    # Create env and run some simulations to evaluate
    print("-- Env Params --")
    print("Gravity: ", grav, " Wind Power: ,", wp)
    env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)    # Create env of this type
    rewards = []
    print("\n ---- Simulation Results (", str(num_tests)," Runs)---- ")
    for i in range(num_tests):
        reward = simulate(env, elite, seed)
        print(str(i) + ". Total Reward: ", reward)
        rewards.append(reward)
    print("Avg Reward: ", sum(rewards)/len(rewards))

if __name__ == "__main__":
    main()