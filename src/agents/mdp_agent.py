import numpy as np
from collections import defaultdict


class MDPAgent:
    """
    Markov Decision Process Agent for scheduling optimization.
    Uses value iteration to compute optimal policy.
    """
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.discount_factor = config.get('discount_factor', 0.9)
        self.learning_rate = config.get('learning_rate', 0.1)
        # Support both 'epsilon' and 'exploration_rate' for consistency
        self.exploration_rate = config.get('epsilon', config.get('exploration_rate', 0.1))
        
        # Value function and policy
        self.value_table = defaultdict(float)
        self.policy = {}
        
        # Transition model (learned from experience)
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Reward model
        self.rewards = defaultdict(lambda: defaultdict(float))
        self.reward_counts = defaultdict(lambda: defaultdict(int))
        
        # Step counter for periodic value iteration
        self._step_counter = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
        
        Returns:
            int: Selected action
        """
        try:
            # Get valid actions from environment
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions:
                # No valid actions, return random action
                return np.random.randint(0, self.env.action_space_size)
            
            # Epsilon-greedy exploration
            if np.random.random() < self.exploration_rate:
                # Explore: random valid action
                return np.random.choice(valid_actions)
            else:
                # Exploit: best action according to value function
                state_key = self._state_to_key(state)
                
                if state_key not in self.policy or self.policy[state_key] not in valid_actions:
                    # If no policy or policy action is invalid, choose best valid action
                    action_values = []
                    for action in valid_actions:
                        try:
                            value = self._get_action_value(state_key, action)
                            action_values.append((action, value))
                        except Exception as e:
                            print(f"Error getting action value: {e}")
                            action_values.append((action, 0.0))
                    
                    if action_values:
                        best_action = max(action_values, key=lambda x: x[1])[0]
                        return best_action
                    else:
                        return np.random.choice(valid_actions)
                
                return self.policy[state_key]
        except Exception as e:
            print(f"Error in select_action: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to random action
            return np.random.randint(0, self.env.action_space_size)
    
    def _get_action_value(self, state_key, action):
        """Calculate Q-value for state-action pair"""
        if state_key not in self.transitions or action not in self.transitions[state_key]:
            return 0.0
        
        # Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * V(s')
        expected_reward = self.rewards[state_key].get(action, 0.0)
        expected_value = 0.0
        
        for next_state_key, prob in self.transitions[state_key][action].items():
            expected_value += prob * self.value_table[next_state_key]
        
        return expected_reward + self.discount_factor * expected_value
    
    def learn(self, state, action, reward, next_state, done=False):
        """
        Learn from experience (alias for update method for consistency with RLAgent).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        try:
            self.update(state, action, reward, next_state, done)
            
            # Periodically run value iteration to improve policy
            # Run value iteration every 50 steps (less frequent to avoid hanging)
            self._step_counter += 1
            if self._step_counter % 50 == 0:
                # Run lightweight value iteration
                self.value_iteration(max_iterations=3, theta=0.1)
        except Exception as e:
            print(f"Error in MDP learn: {e}")
            import traceback
            traceback.print_exc()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the MDP model with new experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Update transition model
        self.transition_counts[state_key][action][next_state_key] += 1
        total_count = sum(self.transition_counts[state_key][action].values())
        
        for next_s, count in self.transition_counts[state_key][action].items():
            self.transitions[state_key][action][next_s] = count / total_count
        
        # Update reward model (running average)
        self.reward_counts[state_key][action] += 1
        count = self.reward_counts[state_key][action]
        old_reward = self.rewards[state_key][action]
        self.rewards[state_key][action] = old_reward + (reward - old_reward) / count
    
    def value_iteration(self, max_iterations=100, theta=1e-4):
        """
        Perform value iteration to compute optimal value function and policy.
        
        Args:
            max_iterations: Maximum number of iterations
            theta: Convergence threshold
        """
        try:
            for iteration in range(max_iterations):
                delta = 0
                
                # Update value for all visited states
                states_to_update = list(self.value_table.keys())
                if not states_to_update:
                    # No states yet, skip value iteration
                    return
                
                for state_key in states_to_update:
                    if not self.transitions[state_key]:
                        continue
                    
                    old_value = self.value_table[state_key]
                    
                    # Compute max Q-value over all actions
                    action_values = []
                    for action in self.transitions[state_key].keys():
                        try:
                            q_value = self._get_action_value(state_key, action)
                            action_values.append((action, q_value))
                        except Exception as e:
                            print(f"Error computing Q-value for state {state_key}, action {action}: {e}")
                            continue
                    
                    if action_values:
                        best_action, best_value = max(action_values, key=lambda x: x[1])
                        self.value_table[state_key] = best_value
                        self.policy[state_key] = best_action
                        
                        delta = max(delta, abs(old_value - best_value))
                
                # Check convergence
                if delta < theta:
                    # print(f"Value iteration converged after {iteration + 1} iterations")
                    break
        except Exception as e:
            print(f"Error in value_iteration: {e}")
            import traceback
            traceback.print_exc()
    
    def policy_evaluation(self, max_iterations=100):
        """Evaluate current policy"""
        for _ in range(max_iterations):
            delta = 0
            for state_key in list(self.policy.keys()):
                old_value = self.value_table[state_key]
                action = self.policy[state_key]
                new_value = self._get_action_value(state_key, action)
                self.value_table[state_key] = new_value
                delta = max(delta, abs(old_value - new_value))
            
            if delta < 1e-4:
                break
    
    def _state_to_key(self, state):
        """Convert state array to hashable key"""
        if isinstance(state, np.ndarray):
            # Discretize continuous state for hashing
            discretized = np.round(state, decimals=2)
            return tuple(discretized)
        return state
    
    def train(self, num_episodes=100):
        """
        Train the MDP agent through interaction with environment.
        
        Args:
            num_episodes: Number of training episodes
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 100:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Periodically update policy
            if (episode + 1) % 10 == 0:
                self.value_iteration(max_iterations=20)
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        # Final policy optimization
        self.value_iteration(max_iterations=100)
        
        return episode_rewards
    
    def save_policy(self, filepath):
        """Save the learned policy to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'policy': dict(self.policy),
                'value_table': dict(self.value_table),
                'transitions': dict(self.transitions),
                'rewards': dict(self.rewards)
            }, f)
    
    def load_policy(self, filepath):
        """Load a saved policy from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.policy = defaultdict(int, data['policy'])
            self.value_table = defaultdict(float, data['value_table'])
            self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)), data['transitions'])
            self.rewards = defaultdict(lambda: defaultdict(float), data['rewards'])