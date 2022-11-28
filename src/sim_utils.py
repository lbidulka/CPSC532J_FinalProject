import torch
import numpy as np

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
    if identifier == "random":
        pred_grav = np.random.uniform(-10.0, 0.0)
        pred_wp = np.random.uniform(0.0, 20.0)
    elif identifier == "oracle":
        pred_grav = env.gravity
        pred_wp = env.wind_power
    else:
        pred_grav, pred_wp = identify(env, identifier, surv_trajectory)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    elite, elite_score = retrieve_elite(elite_archive_df, pred_grav, pred_wp, action_dim, obs_dim)
    elite_reward = sim_elite(env, elite, surv_end_state, surv_step_count)
    total_reward = surv_reward + elite_reward
    return total_reward, pred_grav, pred_wp

# Run the generalist agent, with no survivor or identifier
def sim_generalist_episode(env, seed, policy, render=False):
    # Log rewards
    fitness = 0
    # Run each episode till termination
    terminal = False
    frame_count = 0
    state = env.reset(seed=seed)[0]
    while not terminal:
        if render:
            env.render()
        # Sample action from policy
        probs = policy(torch.from_numpy(state))
        action = torch.argmax(probs)

        # Take action and observe environment
        state, reward, terminal, _, _ = env.step(action.numpy())
        frame_count += 1
        # Log
        fitness += reward
        # Early stopping for runs
        if frame_count > 1000:
            break
    return fitness

# Loads an elites policy from dataframe
def retrieve_elite(df, grav, wp, action_dim, obs_dim):
    df_dists = np.sqrt(np.power(df["behavior_0"]-grav,2) + np.power(df["behavior_1"]-wp,2)) 
    idx = df_dists.argsort()[:1]
    df_closest = df.iloc[idx] 
    elite_score = df_closest["objective"]
    elite = df_closest.loc[:, "solution_0":"solution_31"].to_numpy().reshape((action_dim, obs_dim))         # Extract model params

    return elite, elite_score
