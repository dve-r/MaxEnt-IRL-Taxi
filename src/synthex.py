"""
Synthetic Expert Generator for Taxi-v3

This module engineers a handcrafted reward function using state-action features
and performs Value Iteration to generate an optimal "expert" policy. 
It then simulates the policy and exports expert trajectories for use in 
Inverse Reinforcement Learning (IRL).
"""

import gymnasium as gym
import numpy as np

def manhattan_dist(loc1, loc2):
    """
    Calculates the Manhattan (L1) distance between two locations.

    Args:
        loc1 (tuple or list): The (x, y) coordinates of the first location.
        loc2 (tuple or list): The (x, y) coordinates of the second location.

    Returns:
        float: The Manhattan distance.
    """
    p1 = np.array(loc1)
    p2 = np.array(loc2)
    return np.linalg.norm(p1 - p2, ord=1)

def generate_trajectories(env, policy, num_trajectories=200):
    """
    Generates a set of expert trajectories using the provided policy.

    Args:
        env (gym.Env): The initialized Gymnasium environment.
        policy (np.ndarray): The expert policy array mapping states to actions.
        num_trajectories (int): The number of episodes to record. Defaults to 200.

    Returns:
        np.ndarray: An array of trajectories, where each trajectory is a list 
                    of (state, action) tuples. Returned as an object array due 
                    to variable trajectory lengths.
    """
    expert_trajectories = []
    for _ in range(num_trajectories):
        state, _ = env.reset()
        done = False
        trajectory = [] # List stores (state, action) pairs

        while not done:
            action = policy[state]
            trajectory.append((state, action))
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if len(trajectory) > 200: # Failsafe for infinite loops
                break
        
        expert_trajectories.append(np.array(trajectory, dtype=int))

    return np.array(expert_trajectories, dtype=object)

def engineered_features(env, state_id, action):
    """
    Extracts high-level features from the environment state and chosen action.

    Args:
        env (gym.Env): The initialized Gymnasium environment.
        state_id (int): The discrete state identifier.
        action (int): The discrete action chosen.

    Returns:
        list: A list of 5 feature values: [dropoff, pickup, dist_pass, dist_dest, step_cost].
    """
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state_id)
    taxi_loc = np.array([taxi_row, taxi_col])
    locations = np.array([(0,0), (0,4), (4,0), (4,3)])
    dest_loc = locations[dest_idx]

    # Passenger location logic
    if pass_idx < 4:
        pass_loc = locations[pass_idx]
        pass_in_taxi = False
    else:
        pass_loc = taxi_loc
        pass_in_taxi = True

    # Feature 1: Successful dropoff
    reached_dest = np.array_equal(taxi_loc, dest_loc)
    dropoff = 1.0 if (action == 5 and pass_in_taxi and reached_dest) else 0.0

    # Feature 2: Successful pickup
    reached_pass = np.array_equal(taxi_loc, pass_loc)
    pickup = 1.0 if (action == 4 and not pass_in_taxi and reached_pass) else 0.0

    # Feature 3: Negative Distance to Passenger
    dist_pass = -1.0 * manhattan_dist(taxi_loc, pass_loc)

    # Feature 4: Negative Distance to Destination
    dist_dest = -1.0 * manhattan_dist(taxi_loc, dest_loc)

    # Feature 5: Constant Action Cost
    step_cost = 1.0

    return [dropoff, pickup, dist_pass, dist_dest, step_cost]


if __name__ == "__main__":
    # 1. Environment Setup
    env = gym.make("Taxi-v3", render_mode='ansi') # 'ansi' is faster for training
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # 2. Ground Truth Weights & Reward Matrix
    W = np.array([21.0, 2.0, 2.0, 2.0, -1.0]) # Fine-tuned weights
    R_table = np.zeros((num_states, num_actions))

    for s in range(num_states):
        for a in range(num_actions):
            state_features = engineered_features(env, s, a)
            engineered_reward = np.dot(W, state_features)
            R_table[s, a] = engineered_reward

    # 3. Value Iteration
    print("\nStarting Value Iteration...")
    gamma = 0.9
    epsilon = 0.1
    v = np.zeros(num_states)
    iteration = 0

    while True:
        delta = 0
        v_new = np.zeros(num_states)

        for s in range(num_states):
            Q_values = np.zeros(num_actions)

            for a in range(num_actions):
                # unwrapped.P returns a list of (prob, next_state, reward, done)
                prob, next_state, _, done = env.unwrapped.P[s][a][0]
                # Bellman Equation
                Q_values[a] = prob * (R_table[s,a] + gamma * v[next_state] * (1-done))

            v_new[s] = np.max(Q_values)
            delta = max(delta, abs(v_new[s] - v[s]))
        
        v = v_new
        iteration += 1
        if delta < epsilon:
            break

    print(f"Converged in {iteration} iterations")

    # 4. Extracting Policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_values_values = np.zeros(num_actions)
        for a in range(num_actions):
            prob, next_state, _, done = env.unwrapped.P[s][a][0]
            Q_values_values[a] = prob * (R_table[s,a] + gamma * v[next_state] * (1 - done))
        policy[s] = np.argmax(Q_values_values)

    # 5. Run Simulation to Validate
    env_sim = gym.make("Taxi-v3", render_mode='human')
    total_rewards = []
    total_steps = []

    print("\nRunning Simulation...")
    for episode in range(1, 11): # 10 episodes
        state, _ = env_sim.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not terminated and not truncated:
            action = policy[state]
            state, reward, terminated, truncated, _ = env_sim.step(action)
            episode_reward += reward
            steps += 1

        print(f"Episode {episode} Result: {steps} steps, Score: {episode_reward}")
        total_rewards.append(episode_reward) 
        total_steps.append(steps)

    env_sim.close()

    print(f"\nAverage Reward: {np.mean(total_rewards)}")
    print(f"Average Steps:  {np.mean(total_steps)}")

    # 6. Generate and Save Trajectories
    print("\nGenerating Expert Trajectories...")
    env_data_gen = gym.make("Taxi-v3", render_mode='ansi')
    trajectories = generate_trajectories(env_data_gen, policy, num_trajectories=200)

    # Save to disk
    np.save("expert_trajectories.npy", trajectories, allow_pickle=True)
    print("Saved to 'expert_trajectories.npy'")