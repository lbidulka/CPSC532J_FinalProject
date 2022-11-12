import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json
from pathlib import Path

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

import torch
from torch import nn
import torch.nn.functional as F

""" CITATION:
@article{pyribs_lunar_lander,
  title   = {Using CMA-ME to Land a Lunar Lander Like a Space Shuttle},
  author  = {Bryon Tjanaka and Sam Sommerer and Nikitas Klapsis and Matthew C. Fontaine and Stefanos Nikolaidis},
  journal = {pyribs.org},
  year    = {2021},
  url     = {https://docs.pyribs.org/en/stable/tutorials/lunar_lander.html}
}
"""

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Observation space size 8: lander x/y coords, x/y linear velocities, angle, angular velocity, 
        # one boolean representing whether each leg is in contact with ground or not
        #
        # Output: 1 of 4 discrete actions
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        return x

def save_heatmap(archive, filename):
    """Saves a heatmap of the optimizer's archive to the filename.

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=-300, vmax=300, ax=ax)
    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel("Impact y-velocity")
    ax.set_xlabel("Impact x-position")
    fig.savefig(filename)


def save_metrics(outdir, metrics):
    """Saves metrics to png plots and a JSON file.

    Args:
        outdir (Path): output directory for saving files.
        metrics (dict): Metrics as output by run_search.
    """
    # Plots.
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics[metric]["x"], metrics[metric]["y"])
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        fig.savefig(str(outdir / f"{metric.lower().replace(' ', '_')}.png"))

    # JSON file.
    with (outdir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

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
    # if seed is not None:
    #     env.seed(seed)

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []
    obs = env.reset(seed=seed)
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
    # Setup ----------------------------------------
    env = gym.make("LunarLander-v2", enable_wind=True)
    seed = 123
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Grid archive stores solutions (models) in a rectangular grid
    archive = GridArchive(
        [5, 5],  # 5 bins in each dimension.
        [(-1.0, 1.0), (-3.0, 0)],  # (-1, 1) for x-pos and (-3, 0) for y-vel.
    )

    # Improvement emitter uses CMA-ES to search for policies which add new entries to the archive or improve existing ones
    initial_model = np.zeros((action_dim, obs_dim))
    emitters = [
        GaussianEmitter(
            archive,
            initial_model.flatten(),
            1.0,  # Initial step size.
            batch_size=3,
        ) for _ in range(5)  # Create 5 separate emitters.
        # ImprovementEmitter(
        #     archive,
        #     initial_model.flatten(),
        #     1.0,  # Initial step size.
        #     batch_size=3,
        # ) for _ in range(5)  # Create 5 separate emitters.
    ]

    # Optimizer connects archive and emitter together
    optimizer = Optimizer(archive, emitters)

    # QD Search -------------------------------------
    start_time = time.time()
    total_itrs = 500

    for itr in tqdm(range(1, total_itrs + 1)):
        # Request models from the optimizer.
        sols = optimizer.ask()

        # Evaluate the models and record the objectives and BCs.
        objs, bcs = [], []
        grav = -3.0    # default: -10.0
        wp = 0.0        # default: 0.0
        for model in sols:
            env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp)
            obj, impact_x_pos, impact_y_vel = simulate(env, model, seed)
            objs.append(obj)
            bcs.append([impact_x_pos, impact_y_vel])

        # Send the results back to the optimizer.
        optimizer.tell(objs, bcs)

        # Logging.
        if itr % 25 == 0:
            elapsed_time = time.time() - start_time
            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
            print(f"  - Archive Size: {len(archive)}")
            print(f"  - Max Score: {archive.stats.obj_max}")
    
    # Save ------------------------------------------
    outdir = "./CPSC532J_FinalProject/lunar_lander_MAP_outputs"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    optimizer.archive.as_pandas().to_csv(outdir / "archive.csv")
    save_heatmap(optimizer.archive, str(outdir / "heatmap.png"))

    # Plotting --------------------------------------
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=-300, vmax=300)
    plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
    plt.ylabel("Impact y-velocity")
    plt.xlabel("Impact x-position")
    plt.show()

    # Show some example trajectories ----------------
    seed = 123
    grav = -3.0    # default: -10.0
    wp = 0.0        # default: 0.0
    env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, gravity=grav, wind_power=wp)

    # elite = archive.elite_with_behavior([-0.4, -0.50]) # Choose a behaviour which impacts on left
    # for i in range(10):
    #     simulate(env, elite.sol, seed)

    # Run some demos
    num_tests = 10
    rewards = []
    elite = archive.elite_with_behavior([0, 0]) # Choose a behaviour which comes straight down
    for i in range(num_tests):
        reward, _, _ = simulate(env, elite.sol, seed)
        print("Total Reward: ", reward)
        rewards.append(reward)
    print("Avg Reward: ", sum(rewards)/len(rewards))

if __name__ == "__main__":
    main()