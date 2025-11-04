import numpy as np
import random
from collections import deque


class RLAgent:
    """
    Reinforcement Learning Agent using Q-Learning with experience replay.
    Can be extended to use Deep Q-Network (DQN) for larger state spaces.
    """
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.state_size = env.state_space_size
        self.action_size = env.action_space_size
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.exploration_strategy = config.get('exploration_strategy', 'epsilon_greedy')
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Q-table or Q-network
        self.use_deep_q = self.state_size > 1000  # Use DQN for large state spaces
        
        if self.use_deep_q:
            self.q_network = self._build_q_network()
            self.target_network = self._build_q_network()
            self.update_target_network()
        else:
            # Use tabular Q-learning for smaller state spaces
            self.q_table = {}
    
    def _build_q_network(self):
        """Build a neural network for Q-learning (requires TensorFlow/PyTorch)"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(self.action_size, activation='linear')
            ])
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                         loss='mse')
            
            return model
        except ImportError:
            print("TensorFlow not available. Using tabular Q-learning instead.")
            self.use_deep_q = False
            self.q_table = {}
            return None
    
    def update_target_network(self):
        """Update target network with weights from main network"""
        if self.use_deep_q and self.q_network and self.target_network:
            self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy or other exploration strategy.
        
        Args:
            state: Current state vector
        
        Returns:
            int: Selected action
        """
        # Get valid actions
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            return np.random.randint(0, self.action_size)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: best valid action
            if self.use_deep_q:
                state_tensor = np.array([state])
                q_values = self.q_network.predict(state_tensor, verbose=0)[0]
                # Mask invalid actions
                masked_q = np.full(self.action_size, -np.inf)
                masked_q[valid_actions] = q_values[valid_actions]
                return np.argmax(masked_q)
            else:
                state_key = self._state_to_key(state)
                if state_key not in self.q_table:
                    return np.random.choice(valid_actions)
                
                q_values = self.q_table[state_key]
                # Choose best valid action
                best_action = max(valid_actions, key=lambda a: q_values.get(a, 0))
                return best_action
    
    def learn(self, state, action, reward, next_state, done=False):
        """
        Update Q-values based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Learn from experience replay
        if len(self.memory) >= self.batch_size:
            self._replay()
    
    def _replay(self):
        """Learn from batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        if self.use_deep_q:
            self._replay_deep_q(minibatch)
        else:
            self._replay_tabular_q(minibatch)
    
    def _replay_deep_q(self, minibatch):
        """Experience replay for Deep Q-Network"""
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(len(minibatch)):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.discount_factor * np.max(next_q_values[i])
            
            current_q_values[i][actions[i]] = target
        
        # Train network
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
    
    def _replay_tabular_q(self, minibatch):
        """Experience replay for tabular Q-learning"""
        for state, action, reward, next_state, done in minibatch:
            state_key = self._state_to_key(state)
            next_state_key = self._state_to_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # Q-learning update
            if done:
                target = reward
            else:
                next_q_values = self.q_table.get(next_state_key, {})
                max_next_q = max(next_q_values.values()) if next_q_values else 0
                target = reward + self.discount_factor * max_next_q
            
            # Update Q-value
            old_q = self.q_table[state_key].get(action, 0)
            self.q_table[state_key][action] = old_q + self.learning_rate * (target - old_q)
    
    def _state_to_key(self, state):
        """Convert state to hashable key for Q-table"""
        if isinstance(state, np.ndarray):
            discretized = np.round(state, decimals=2)
            return tuple(discretized)
        return state
    
    def train(self, num_episodes=1000, verbose=True):
        """
        Train the RL agent.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress
        
        Returns:
            list: Episode rewards
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
                
                self.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target network periodically
            if self.use_deep_q and (episode + 1) % 10 == 0:
                self.update_target_network()
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def save_model(self, filepath):
        """Save the learned model"""
        if self.use_deep_q and self.q_network:
            self.q_network.save(filepath)
        else:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.q_table, f)
    
    def load_model(self, filepath):
        """Load a saved model"""
        if self.use_deep_q:
            import tensorflow as tf
            self.q_network = tf.keras.models.load_model(filepath)
            self.target_network = tf.keras.models.load_model(filepath)
        else:
            import pickle
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)