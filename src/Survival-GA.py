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

# Policy Network
class GANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

def run_episode(env, policy, seed, render=False):
    # Log rewards
    fitness = 0
    # Run each episode till termination
    terminal = False
    frame_count = 0
    state, _ = env.reset(seed=seed)
    step_limit = 75
    while frame_count < step_limit:
        if render:
            env.render()
        # Sample action from policy
        probs = policy(torch.from_numpy(state))
        action = torch.argmax(probs)

        # Take action and observe environment
        state, reward, terminal, _, _ = env.step(action.numpy())
        # Penalize Eucl. distance from starting point
        sp_dist = np.linalg.norm(state[:2] - [0, 1.4], 2)   
        fitness -= sp_dist  

        # Penalize any final angular velocity
        if frame_count + 5 >= step_limit:
            ang_vels = state[4:6]
            vel_penalty = np.abs(ang_vels).sum()
            fitness -= vel_penalty

        frame_count += 1
    env.close()
    return fitness
    

def train(args, env):
    avg_fitnesses = []
    max_fitnesses = []
    norm = torch.distributions.Normal(0, args.mu_noise)

    # Setup ----------------------------------------
    env = gym.make("LunarLander-v2", enable_wind=True)
    seed = 123
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    # Create initial population
    pop = []
    pop_fitness = torch.zeros(args.pop_size)
    for i in range(args.pop_size):
        pop.append(GANet())

    # Train for a number of episodes
    num_avgruns = 3
    for generation in range(args.GN):
        print("Gen: ", str(generation))
        rewards = []
        new_pop = []
        # Test members of the population, averaged over a few simulations
        for i in tqdm(range(len(pop))):
            temp_fitness = []
            for j in range(num_avgruns):
                temp_fitness.append(run_episode(env, pop[i], render=False, seed=seed))
            pop_fitness[i] = sum(temp_fitness)/len(temp_fitness)
        # Record average performance of population
        avg_fitness = pop_fitness.mean()
        print("Avg fitness: ", avg_fitness)
        avg_fitnesses.append(float(np.sum(avg_fitness.numpy())))

        # After episode, pass on top members
        values, idx = torch.topk(F.softmax(pop_fitness, dim=0), args.num_elites, sorted=True)
        max_fitnesses.append(float(pop_fitness.numpy()[idx[0]]))
        for i in idx:
            new_pop.append(pop[i])

        # Create offspring from each of the elites
        for i in range(args.num_elites):
            elite = new_pop[i]

            for j in range(args.num_offspring):
                offspring = copy.deepcopy(elite)
                # Mutate parameters
                with torch.no_grad():
                    for param in offspring.parameters():
                        param += norm.sample(param.shape)
                new_pop.append(offspring)
                
        pop = new_pop
        pop_fitness = torch.zeros(len(pop))

    torch.save(new_pop[0], "./CPSC532J_FinalProject/src/models/GA-policy.pth")
    return avg_fitnesses, max_fitnesses

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GN", default=30, help="num generations", type=int)
    parser.add_argument("--pop_size", default=150, help="initial population size", type=int)
    parser.add_argument("--num_elites", default=3, help="number of elites saved per episode", type=int)
    parser.add_argument("--num_offspring", default=30, help="number of offspring generated from each elite per episode", type=int)
    parser.add_argument("--mu_noise", default=0.5, help="std dev of mutation noise distribution", type=float)
    return parser.parse_args()


def main():
    input_args = get_args()
    # Log input arguments
    print(os.getcwd())
    date_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = "./CPSC532J_FinalProject/src/logs/GA/" + date_time 
    os.mkdir(log_dir)
    with open(log_dir + "/input_args.json", "w") as fp:
        json.dump(vars(input_args), fp)

    # Setup env
    env = gym.make("LunarLander-v2")
    env.reset()

    # Setup networks and train
    avg_rewards, max_rewards = train(input_args, env)

    # Log and plot results
    with open(log_dir + "/avg_rewards.json", "w") as fp:
        json.dump(avg_rewards, fp)
    with open(log_dir + "/max_rewards.json", "w") as fp:
        json.dump(max_rewards, fp)
    plt.plot(np.arange(0,len(avg_rewards), 1), avg_rewards, label="avg")
    plt.plot(np.arange(0,len(max_rewards), 1), max_rewards, label="max")
    plt.legend()
    plt.title("GA Rewards per Episode")
    plt.xlabel("Episode No.")
    plt.ylabel("Reward")
    plt.savefig(log_dir + "/rewards_plot.png")
    plt.show()    

if __name__ == "__main__":
    main()