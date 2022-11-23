import gym
import numpy as np
from tqdm import tqdm
import torch

from models import Survivor


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

    total_reward = 0.0
    state = env.reset(seed=seed)
    state = state[0]
    done = False

    step_count = 0
    step_limit = 50

    trajectory = []

    while not done:
        probs = model(torch.from_numpy(state))
        action = torch.argmax(probs)
        state, reward, done, info, _ = env.step(action.numpy())
        step_count += 1
        total_reward += reward

        trajectory.append(np.concatenate((state, action.numpy().reshape((-1,)), reward.reshape((-1,)))))

        if step_count >= step_limit:
            break

    return trajectory

def main():
    seed = 123

    # Load survivor 
    elite = torch.load("./CPSC532J_FinalProject/src/model_checkpoints/Survivor-policy.pth")
    elite.eval()

    # Run some demos
    num_tests = 250000
    
    print("Running sims...")
    trajectories = []
    labels = []
    for i in tqdm(range(num_tests)):
        # Random sample env parameters from uniform distribution
        grav = np.random.uniform(-10.0, 0.0)
        wp = np.random.uniform(0.0, 20.0)
        env = gym.make("LunarLander-v2", enable_wind=True, gravity=grav, wind_power=wp)
        trajectory = simulate(env, elite, seed)
        trajectories.append(trajectory)
        labels.append((grav, wp))
    
    print("Saving data...")
    np_trajectories = np.stack(trajectories)
    np_labels = np.stack(labels)
    np.save("./CPSC532J_FinalProject/data/trajectories.npy", np_trajectories)
    np.save("./CPSC532J_FinalProject/data/labels.npy", np_labels)

if __name__ == "__main__":
    main()