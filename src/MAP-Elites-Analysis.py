import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
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
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
    """

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []
    obs = env.reset(seed=123)
    obs = obs[0]
    done = False

    step_count = 0

    while not done:
        action = np.argmax(model @ obs)  # Linear policy.
        obs, reward, done, info, _ = env.step(action)
        step_count += 1
        total_reward += reward

        # Refer to the definition of state here:
        # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L306
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        if step_count >= 1000:
            break

    # If the lunar lander did not land, set the x-pos to the one from the final
    # timestep, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)

    return total_reward, impact_x_pos, impact_y_vel

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
    el_wp = 15.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------
    # Env Parameters ---------------------------------
    grav = -10.0    # default: -10.0, range: [-10, 0]
    wp = 15.0        # default: 0.0, range: [0, 20]
    # ------------------------------------------------

    # Get elite for this environment parameter region
    df_dists = np.sqrt(np.power(df["behavior_0"]-el_grav,2) + np.power(df["behavior_1"]-el_wp,2)) 
    idx = df_dists.argsort()[:1]
    df_closest = df.iloc[idx] 
    # print(df_closest)
    print("-- Elite Params --")
    print("Gravity: ", el_grav, " Wind Power: ,", el_wp)
    print("Recorded score: ", df_closest.at[0,"objective"])
    elite = df_closest.loc[:, "solution_0":"solution_31"].to_numpy().reshape((action_dim, obs_dim))         # Extract model params

    # Create env and run some simulations to evaluate
    print("-- Env Params --")
    print("Gravity: ", grav, " Wind Power: ,", wp)
    env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)    # Create env of this type
    rewards = []
    print("\n ---- Simulation Results ---- ")
    for i in range(num_tests):
        reward, _, _ = simulate(env, elite, seed)
        print("Total Reward: ", reward)
        rewards.append(reward)
    print("Avg Reward: ", sum(rewards)/len(rewards))

if __name__ == "__main__":
    main()