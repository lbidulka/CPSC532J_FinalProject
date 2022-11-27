import gym
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
import argparse

import torch
from torch import nn
import torch.nn.functional as F

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# # Policy Network
# class PolicyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(8, 4)
        
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.softmax(x, dim=0)
#         return x
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

# State-value Network
class StateValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

def train(args, env, seed, policy, state_val, policy_optimizer, stateval_optimizer):
    ep_rewards = []

    # Train for a number of episodes
    for episode in tqdm(range(args.EN)):
        state = env.reset(seed=seed)[0]
        terminal = False
        step_count = 0
        total_reward = 0
        G = 0
        # Log states, actions, rewards in episode
        states = []
        actions = []
        logprobs = []
        rewards = []
        # For backprop
        probs = policy(torch.from_numpy(state))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        # Run each episode till termination
        while not terminal:
            if episode % 500 == 0:
                env.render()
            # Sample action from policy
            probs = policy(torch.from_numpy(state))
            m = torch.distributions.Categorical(probs)
            action = m.sample()

            # Take action and observe environment
            state, reward, terminal, _, _ = env.step(action.numpy())
            step_count += 1
            # Log
            states.append(state)
            actions.append(action)
            logprobs.append(m.log_prob(action))
            rewards.append(reward)
            if step_count >= 1000:
                break

        env.close()

        # After episode, do some learning
        returns = []
        pol_losses = []
        stateval_losses = []

        returns = []
        G = 0
        for r in rewards[::-1]:
            G = r + args.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # v, delta
        state_vals = state_val(torch.tensor(np.array(states)))
        deltas = torch.detach(returns - state_vals.view(-1))

        # Losses
        for i, delta in enumerate(deltas):
            pol_loss = -logprobs[i] * delta
            pol_losses.append(torch.reshape(pol_loss, (-1,1)))
            stateval_loss = -state_vals[i] * delta
            stateval_losses.append(torch.reshape(stateval_loss, (-1,1)))

        # Grad update
        policy_optimizer.zero_grad()
        stateval_optimizer.zero_grad()

        stateval_loss = torch.cat(stateval_losses).sum()
        stateval_loss.backward() 
        pol_loss = torch.cat(pol_losses).sum()
        pol_loss.backward() 
        
        policy_optimizer.step()     
        stateval_optimizer.step()        

        ep_rewards.append(np.sum(np.asarray(rewards)))

        # If last 10 iterations all have at least 190 score, we have solved the problem
        if len(ep_rewards) >= 5:
            if (np.asarray(ep_rewards[-5:]) >= 190).all():
                torch.save(policy, "./models/PG-policy.pth")
                return ep_rewards

    torch.save(policy, "./CPSC532J_FinalProject/src/model_checkpoints/PG-policy.pth")
    return ep_rewards

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EN", default=2500, help="num episodes", type=int)
    parser.add_argument("--gamma", default=0.99, help="discount factor", type=float)
    parser.add_argument("--policy_lr", default=1e-3, help="policy lr", type=float)
    parser.add_argument("--stateval_lr", default=1e-6, help="state value lr", type=float)
    parser.add_argument("--policy_lrdecay", default=0.0000, help="policy optimizer decay", type=float)
    parser.add_argument("--stateval_lrdecay", default=0.0000, help="stateval optimizer decay", type=float)
    return parser.parse_args()

def main():
    input_args = get_args()
    # Log input arguments
    date_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = "./CPSC532J_FinalProject/src/logs/PG/" + date_time 
    os.mkdir(log_dir)
    with open(log_dir + "/input_args.json", "w") as fp:
        json.dump(vars(input_args), fp)

    # Setup env
    seed = 123
    env = gym.make("LunarLander-v2")
    env.reset(seed=seed)

    # Setup networks and train
    policy = PolicyNet()
    state_val = StateValueNet()

    # policy_lrs = [1e-3, 1e-4, 1e-5]
    # stateval_lrs = [1e-3, 5e-6, 1e-6, 1e-9]

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=input_args.policy_lr, weight_decay=input_args.policy_lrdecay)
    stateval_optimizer = torch.optim.Adam(state_val.parameters(), lr=input_args.stateval_lr, weight_decay=input_args.stateval_lrdecay)

    rewards = train(input_args, env, seed, policy, state_val, policy_optimizer, stateval_optimizer)

    # Log and plot results
    with open(log_dir + "/ep_rewards.json", "w") as fp:
        json.dump(rewards, fp)
    plt.plot(np.arange(0,len(rewards), 1), rewards, alpha=0.8)
    plt.title("PG Total Reward per Episode")
    plt.xlabel("Episode No.")
    plt.ylabel("Total Reward")
    plt.savefig(log_dir + "/rewards_plot.png")
    plt.show()    

if __name__ == "__main__":
    main()