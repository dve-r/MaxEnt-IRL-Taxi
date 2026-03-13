"""
L2 Regularized Maximum Entropy Inverse Reinforcement Learning

This module implements the MaxEnt IRL algorithm with an added L2 (Ridge) 
Regularization penalty. Regularization helps prevent the reward weights from 
growing unbounded (overfitting) when trying to perfectly match the expert's 
empirical feature expectations.
"""

import gymnasium as gym
import numpy as np
import time

# ==========================================
# Helper Functions
# ==========================================

def manhattan_dist(loc1, loc2):
    """Calculates the Manhattan (L1) distance between two locations."""
    p1 = np.array(loc1)
    p2 = np.array(loc2)
    return np.linalg.norm(p1 - p2, ord=1)

def engineered_features(env, state_id, action):
    """
    Extracts high-level features from the environment state and chosen action.
    
    Returns:
        list: [dropoff, pickup, dist_pass, dist_dest, step_cost]
    """
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state_id)
    taxi_loc = np.array([taxi_row, taxi_col])
    locations = np.array([(0,0), (0,4), (4,0), (4,3)])
    dest_loc = locations[dest_idx]

    # Passenger location
    if pass_idx < 4:
        pass_loc = locations[pass_idx]
        pass_in_taxi = False
    else:
        pass_loc = taxi_loc
        pass_in_taxi = True

    # Feature calculation
    reached_dest = np.array_equal(taxi_loc, dest_loc)
    dropoff = 1.0 if (action == 5 and pass_in_taxi and reached_dest) else 0.0

    reached_pass = np.array_equal(taxi_loc, pass_loc)
    pickup = 1.0 if (action == 4 and not pass_in_taxi and reached_pass) else 0.0

    dist_pass = -1.0 * manhattan_dist(taxi_loc, pass_loc)
    dist_dest = -1.0 * manhattan_dist(taxi_loc, dest_loc)
    step_cost = 1.0

    return [dropoff, pickup, dist_pass, dist_dest, step_cost]

def generate_trajectories(env, policy, num_trajectories=200):
    """Generates expert trajectories using the provided optimal policy."""
    expert_trajectories = []
    for i in range(num_trajectories):
        state, _ = env.reset()
        done = False
        trajectory = [] 

        while not done:
            action = policy[state]
            trajectory.append((state, action))
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if len(trajectory) > 200: # Safety break
                break
        
        expert_trajectories.append(np.array(trajectory, dtype=int))

    return np.array(expert_trajectories, dtype=object)

def visualize_learned_policy(policy, num_episodes=3):
    """Creates a temporary environment to visually render the learned policy."""
    temp_env = gym.make("Taxi-v3", render_mode='human')
    print(f"\nVisualizing learned policy ({num_episodes} Episodes)")
    
    for i in range(num_episodes):
        state, _ = temp_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = temp_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            time.sleep(0.3)
            if steps > 50:
                break
        
        print(f"Episode {i+1} Finished. Steps: {steps}, Total Reward: {total_reward}")

    temp_env.close()

# ==========================================
# Value Iteration
# ==========================================

def value_iteration(env, R_table, gamma=0.9, epsilon=0.1):
    """
    Performs Value Iteration to find the optimal policy given a reward table.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    v = np.zeros(num_states)
    iteration = 0

    print("Starting Value Iteration...")

    while True:
        delta = 0
        v_new = np.zeros(num_states)

        for s in range(num_states):
            Q_values = np.zeros(num_actions)
            for a in range(num_actions):
                prob, next_state, _, done = env.unwrapped.P[s][a][0]
                Q_values[a] = prob * (R_table[s, a] + gamma * v[next_state] * (1 - done))

            v_new[s] = np.max(Q_values)
            delta = max(delta, abs(v_new[s] - v[s]))
        
        v = v_new
        iteration += 1
        
        if delta < epsilon:
            print(f"Converged in {iteration} iterations")
            break

    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_values = np.zeros(num_actions)
        for a in range(num_actions):
            prob, next_state, _, done = env.unwrapped.P[s][a][0]
            Q_values[a] = prob * (R_table[s, a] + gamma * v[next_state] * (1 - done))
        
        policy[s] = np.argmax(Q_values)

    return v, policy

# ==========================================
# L2 Regularized MaxEnt IRL Core Functions
# ==========================================

def precompute_feature_matrix(env, num_states, num_actions):
    """Precomputes the feature vector for every (state, action) pair."""
    feat_dim = len(engineered_features(env, 0, 0)) 
    feature_matrix = np.zeros((num_states, num_actions, feat_dim))

    for s in range(num_states):
        for a in range(num_actions):
            feature_matrix[s, a] = engineered_features(env, s, a)
    
    return feature_matrix

def get_expert_feature_expectations(feature_matrix, trajectories):
    """Computes the empirical mean feature count over all expert trajectories."""
    feature_expectations = np.zeros(feature_matrix.shape[2])
    
    for traj in trajectories:
        for state, action in traj:
            feature_expectations += feature_matrix[state, action]
            
    feature_expectations /= len(trajectories)
    return feature_expectations

def maxent_irl(env, feature_matrix, expert_expectations, start_probs=None, 
               n_iterations=100, lr=0.01, horizon=50, l2_penalty=0.1):
    """
    Executes the Maximum Entropy IRL algorithm with L2 regularization.
    
    Args:
        env (gym.Env): The underlying environment.
        feature_matrix (np.ndarray): Precomputed state-action features.
        expert_expectations (np.ndarray): Empirical feature counts from expert.
        start_probs (np.ndarray, optional): Initial state distribution.
        n_iterations (int): Number of gradient descent steps.
        lr (float): Learning rate.
        horizon (int): Time horizon for the forward/backward passes.
        l2_penalty (float): The regularization strength (lambda).
        
    Returns:
        np.ndarray: The recovered reward weights (theta).
    """
    num_states, num_actions, num_feats = feature_matrix.shape
    theta = np.random.uniform(size=(num_feats,))

    if start_probs is None:
        start_probs = np.ones(num_states) / num_states

    for it in range(n_iterations):
        rewards = np.dot(feature_matrix, theta) # Shape: (S, A)
        
        # --- Backward Pass (Log Space) ---
        Zs = np.zeros((horizon + 1, num_states))
        Qs = np.zeros((horizon, num_states, num_actions))
        
        for t in range(horizon - 1, -1, -1):
            for s in range(num_states):
                for a in range(num_actions):
                    log_probs = []
                    v_next_vals = []
                    
                    for prob, next_state, _, _ in env.unwrapped.P[s][a]:
                        if prob > 0:
                            log_probs.append(np.log(prob))
                            v_next_vals.append(Zs[t+1, next_state])
                    
                    if not log_probs:
                        log_expected_next_z = -1e10
                    else:
                        terms = np.array(log_probs) + np.array(v_next_vals)
                        max_term = np.max(terms)
                        log_expected_next_z = max_term + np.log(np.sum(np.exp(terms - max_term)))
                        
                    Qs[t, s, a] = rewards[s, a] + log_expected_next_z
                
                max_q = np.max(Qs[t, s, :])
                Zs[t, s] = max_q + np.log(np.sum(np.exp(Qs[t, s, :] - max_q)))

        policy = np.exp(Qs - Zs[:-1, :, None]) # Shape: (Horizon, S, A)

        # --- Forward Pass (State Visitation Frequencies) ---
        D = np.zeros((horizon + 1, num_states))
        D[0, :] = start_probs
        expected_svf_sa = np.zeros((num_states, num_actions))
        
        for t in range(horizon):
            for s in range(num_states):
                if D[t, s] < 1e-10: continue 
                
                for a in range(num_actions):
                    p_action = policy[t, s, a]
                    flow = D[t, s] * p_action
                    expected_svf_sa[s, a] += flow
                    
                    for prob, next_state, _, done in env.unwrapped.P[s][a]:
                        if not done:
                            D[t+1, next_state] += flow * prob

        learner_expectations = np.sum(expected_svf_sa[:, :, None] * feature_matrix, axis=(0, 1))
        
        # --- Gradient Update with L2 Regularization ---
        # Note: We subtract (l2_penalty * theta) to penalize large weights
        grad = expert_expectations - learner_expectations - (l2_penalty * theta)
        theta += lr * grad
        
        diff = np.linalg.norm(expert_expectations - learner_expectations)
        if it % 5 == 0:
            print(f"Iteration {it:02d} | Feature Diff Norm: {diff:.4f}")

    return theta

# ==========================================
# Main Execution Block
# ==========================================

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode='ansi')
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    W = np.array([21.0, 2.0, 2.0, 2.0, -1.0]) 

    print("Generating Expert Policy using Value Iteration...")
    R_table = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            R_table[s, a] = np.dot(W, engineered_features(env, s, a))

    optimal_values, optimal_policy = value_iteration(env, R_table, gamma=0.9, epsilon=0.1)

    feature_matrix = precompute_feature_matrix(env, num_states, num_actions)
    expert_trajs = generate_trajectories(env, optimal_policy, num_trajectories=200)
    expert_feat_exp = get_expert_feature_expectations(feature_matrix, expert_trajs)

    start_counts = np.zeros(num_states)
    for traj in expert_trajs:
        start_counts[traj[0][0]] += 1
    start_probs = start_counts / len(expert_trajs)

    print("\nRunning L2-Regularized MaxEnt IRL...")
    # Using L2 penalty of 0.05
    recovered_weights = maxent_irl(env, feature_matrix, expert_feat_exp, start_probs=start_probs, 
                                   n_iterations=100, lr=0.01, horizon=50, l2_penalty=0.05)

    print("\n=== Results ===")
    print("Ground Truth Weights: ", W)
    print("Recovered Weights:    ", recovered_weights)
    
    print("\nNormalized Comparison (Divided by max abs value):")
    print("GT (Norm):", np.round(W / np.max(np.abs(W)), 2))
    print("RW (Norm):", np.round(recovered_weights / np.max(np.abs(recovered_weights)), 2))

    # --- Testing ---
    print("\nTesting learned weights by deriving new policy...")
    Learned_R_table = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            Learned_R_table[s, a] = np.dot(recovered_weights, engineered_features(env, s, a))

    _, learned_policy = value_iteration(env, Learned_R_table, gamma=0.9, epsilon=0.1)

    diff = np.sum(learned_policy != optimal_policy)
    print(f"Policy Disagreement: The learned policy differs from expert in {diff} states.")

    visualize_learned_policy(learned_policy)