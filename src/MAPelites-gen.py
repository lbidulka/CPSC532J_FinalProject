import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path

from CPSC532J_FinalProject.src.models import Survivor
from sim_utils import sim_elite, sim_survivor

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

def save_heatmap(archive, filename):
    """Saves a heatmap of the optimizer's archive to the filename.

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=-300, vmax=300, ax=ax)
    # ax.invert_xaxis()  # Makes more sense if larger gravities are on top.
    ax.set_xlabel("Gravity")
    ax.set_ylabel("Wind Power")
    fig.savefig(filename)
    fig.show()


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

def main():
    # Setup ----------------------------------------
    env = gym.make("LunarLander-v2", enable_wind=True)
    seed = 123
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Load survivor model
    survivor = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/Survivor-policy.pth")
    survivor.eval()

    # Grid archive stores solutions (models) in a rectangular grid
    archive = GridArchive(
        [7, 7],  # 5 bins in each dimension.
        [(-10.0, 0.0), (0.0, 20.0)],  # (-10, 0) for gravity and (0, 20) for wind power.
    )

    # GAUSS EMITTER IS GOOD HERE.... explain why
    initial_model = np.zeros((action_dim, obs_dim))
    emitters = [
        GaussianEmitter(
            archive,
            initial_model.flatten(),
            1.0,  # Initial step size.
            batch_size=10,
        ) for _ in range(5)  # Create 5 separate emitters.
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
        for model in sols:
            # Random sample env parameters from uniform distribution
            grav = np.random.uniform(-10.0, 0.0)
            wp = np.random.uniform(0.0, 20.0)
            # We will average the model on a few simulations, to enforce some policy robustness
            avging_runs = 10
            mod_objs = []
            for i in range(avging_runs):
                env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp)
                surv_reward, surv_trajectory, surv_end_state, surv_step_count = sim_survivor(env, survivor, seed)
                elite_reward = sim_elite(env, model, surv_end_state, surv_step_count)
                obj = surv_reward + elite_reward
                # obj, impact_x_pos, impact_y_vel = simulate(env, model, seed)
                mod_objs.append(obj)

            # Get avg results
            obj = sum(mod_objs) / len(mod_objs)

            objs.append(obj)
            bcs.append([grav, wp])

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
            print("  - saved MAP.")            
    
    # Save ------------------------------------------
    outdir = "./CPSC532J_FinalProject/lunar_lander_MAP_outputs"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    optimizer.archive.as_pandas().to_csv(outdir / "archive.csv")
    save_heatmap(optimizer.archive, str(outdir / "heatmap.png"))

if __name__ == "__main__":
    main()