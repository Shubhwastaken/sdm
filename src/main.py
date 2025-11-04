import yaml
import numpy as np
import os
from environment.scheduling_env import SchedulingEnvironment
from agents.mdp_agent import MDPAgent
from agents.rl_agent import RLAgent
from visualization.plotter import SchedulingPlotter


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_mdp_simulation(env, config, num_episodes=100):
    """
    Run simulation using MDP agent.
    
    Args:
        env: Scheduling environment
        config: Configuration dictionary
        num_episodes: Number of episodes to run
    
    Returns:
        tuple: (agent, episode_rewards, metrics_history)
    """
    print("\n" + "="*80)
    print("RUNNING MDP AGENT SIMULATION")
    print("="*80 + "\n")
    
    mdp_agent = MDPAgent(env, config['agents']['mdp'])
    episode_rewards = []
    metrics_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config['simulation']['max_steps_per_episode']:
            # Select action using MDP agent
            action = mdp_agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update MDP model
            mdp_agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        metrics_history.append(info)
        
        # Periodically update policy and display progress
        if (episode + 1) % 10 == 0:
            mdp_agent.value_iteration(max_iterations=20)
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Scheduled: {info['scheduled_classes']} | "
                  f"Disruptions: {info['disruptions']}")
        
        # Render final episode
        if (episode + 1) % config['simulation']['log_interval'] == 0:
            print(f"\nEpisode {episode + 1} Final State:")
            env.render()
    
    # Final policy optimization
    mdp_agent.value_iteration(max_iterations=100)
    
    return mdp_agent, episode_rewards, metrics_history


def run_rl_simulation(env, config, num_episodes=1000):
    """
    Run simulation using RL agent (Q-Learning/DQN).
    
    Args:
        env: Scheduling environment
        config: Configuration dictionary
        num_episodes: Number of episodes to run
    
    Returns:
        tuple: (agent, episode_rewards, metrics_history)
    """
    print("\n" + "="*80)
    print("RUNNING RL AGENT SIMULATION")
    print("="*80 + "\n")
    
    rl_agent = RLAgent(env, config['agents']['rl'])
    episode_rewards = []
    metrics_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config['simulation']['max_steps_per_episode']:
            # Select action using RL agent
            action = rl_agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Learn from experience
            rl_agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        metrics_history.append(info)
        
        # Decay exploration rate
        if rl_agent.epsilon > rl_agent.epsilon_min:
            rl_agent.epsilon *= rl_agent.epsilon_decay
        
        # Update target network (for DQN)
        if rl_agent.use_deep_q and (episode + 1) % 10 == 0:
            rl_agent.update_target_network()
        
        # Display progress
        if (episode + 1) % config['simulation']['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {rl_agent.epsilon:.3f} | "
                  f"Scheduled: {info['scheduled_classes']} | "
                  f"Disruptions: {info['disruptions']}")
            
            # Render state
            if (episode + 1) % (config['simulation']['log_interval'] * 2) == 0:
                env.render()
    
    return rl_agent, episode_rewards, metrics_history


def run_hybrid_simulation(env, config, num_episodes=100):
    """
    Run hybrid simulation using both MDP and RL agents.
    MDP agent makes initial decisions, RL agent learns to improve.
    
    Args:
        env: Scheduling environment
        config: Configuration dictionary
        num_episodes: Number of episodes to run
    
    Returns:
        tuple: (mdp_agent, rl_agent, episode_rewards, metrics_history)
    """
    print("\n" + "="*80)
    print("RUNNING HYBRID MDP + RL SIMULATION")
    print("="*80 + "\n")
    
    mdp_agent = MDPAgent(env, config['agents']['mdp'])
    rl_agent = RLAgent(env, config['agents']['rl'])
    
    episode_rewards = []
    metrics_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config['simulation']['max_steps_per_episode']:
            # Use MDP for initial phase, switch to RL for optimization
            if episode < num_episodes // 3:
                action = mdp_agent.select_action(state)
            else:
                action = rl_agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Both agents learn
            mdp_agent.update(state, action, reward, next_state, done)
            rl_agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        metrics_history.append(info)
        
        # Update policies
        if (episode + 1) % 10 == 0:
            mdp_agent.value_iteration(max_iterations=10)
        
        # Display progress
        if (episode + 1) % config['simulation']['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            agent_type = "MDP" if episode < num_episodes // 3 else "RL"
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Agent: {agent_type} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Scheduled: {info['scheduled_classes']}")
    
    return mdp_agent, rl_agent, episode_rewards, metrics_history


def visualize_results(mdp_rewards, rl_rewards, mdp_metrics, rl_metrics, env, config):
    """
    Visualize simulation results.
    
    Args:
        mdp_rewards: MDP agent episode rewards
        rl_rewards: RL agent episode rewards
        mdp_metrics: MDP agent metrics history
        rl_metrics: RL agent metrics history
        env: Environment instance
        config: Configuration dictionary
    """
    plotter = SchedulingPlotter()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Plot training progress
    print("\nGenerating visualizations...")
    
    if mdp_rewards:
        plotter.plot_training_progress(mdp_rewards, save_path='output/mdp_training.png')
    
    if rl_rewards:
        plotter.plot_training_progress(rl_rewards, save_path='output/rl_training.png')
    
    # Compare agents
    if mdp_rewards and rl_rewards:
        plotter.plot_comparison(mdp_rewards, rl_rewards, save_path='output/comparison.png')
    
    # Plot metrics
    if mdp_metrics:
        plotter.plot_scheduling_metrics(mdp_metrics, save_path='output/mdp_metrics.png')
    
    if rl_metrics:
        plotter.plot_scheduling_metrics(rl_metrics, save_path='output/rl_metrics.png')
    
    # Plot final schedule
    state = env.reset()
    # Run one episode to get final schedule
    done = False
    step = 0
    while not done and step < 50:
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = np.random.choice(valid_actions)
            state, reward, done, info = env.step(action)
        step += 1
    
    plotter.plot_schedule_heatmap(
        env.schedule,
        config['environment']['num_teachers'],
        config['environment']['num_rooms'],
        env.num_time_slots,
        save_path='output/schedule_heatmap.png'
    )
    
    # Plot disruption analysis
    if env.active_disruptions:
        plotter.plot_disruption_analysis(env.active_disruptions, 
                                        save_path='output/disruptions.png')
    
    print("Visualizations saved to 'output/' directory")


def save_models(mdp_agent, rl_agent, config):
    """Save trained models"""
    os.makedirs('models', exist_ok=True)
    
    if config['simulation'].get('save_model', True):
        try:
            mdp_agent.save_policy('models/mdp_policy.pkl')
            print("MDP policy saved to models/mdp_policy.pkl")
        except Exception as e:
            print(f"Error saving MDP policy: {e}")
        
        try:
            model_path = config['simulation'].get('model_save_path', 'models/rl_agent_model.pkl')
            rl_agent.save_model(model_path)
            print(f"RL model saved to {model_path}")
        except Exception as e:
            print(f"Error saving RL model: {e}")


def main():
    """Main simulation entry point"""
    # Load configuration (adjust path for running from src directory)
    config_path = '../config/simulation_config.yaml' if os.path.exists('../config/simulation_config.yaml') else 'config/simulation_config.yaml'
    config = load_config(config_path)
    
    print("="*80)
    print(" REINFORCEMENT LEARNING SCHEDULING SIMULATOR")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Classes: {config['environment']['num_classes']}")
    print(f"  Teachers: {config['environment']['num_teachers']}")
    print(f"  Rooms: {config['environment']['num_rooms']}")
    print(f"  Disruption Probability: {config['environment']['disruption_probability']}")
    print(f"  Episodes: {config['simulation']['num_episodes']}")
    print("="*80)
    
    # Initialize environment
    env = SchedulingEnvironment(config['environment'])
    
    # Run simulations
    mode = input("\nSelect simulation mode:\n  1. MDP Only\n  2. RL Only\n  3. Hybrid (MDP + RL)\n  4. Compare MDP vs RL\n> ")
    
    mdp_rewards, rl_rewards = None, None
    mdp_metrics, rl_metrics = None, None
    mdp_agent, rl_agent = None, None
    
    if mode == '1':
        mdp_agent, mdp_rewards, mdp_metrics = run_mdp_simulation(
            env, config, num_episodes=config['simulation']['num_episodes']
        )
    elif mode == '2':
        rl_agent, rl_rewards, rl_metrics = run_rl_simulation(
            env, config, num_episodes=config['simulation']['num_episodes']
        )
    elif mode == '3':
        mdp_agent, rl_agent, hybrid_rewards, hybrid_metrics = run_hybrid_simulation(
            env, config, num_episodes=config['simulation']['num_episodes']
        )
        mdp_rewards = hybrid_rewards
        mdp_metrics = hybrid_metrics
    elif mode == '4':
        # Run both for comparison
        mdp_agent, mdp_rewards, mdp_metrics = run_mdp_simulation(
            env, config, num_episodes=config['simulation']['num_episodes']
        )
        env.reset()  # Reset environment
        rl_agent, rl_rewards, rl_metrics = run_rl_simulation(
            env, config, num_episodes=config['simulation']['num_episodes']
        )
    else:
        print("Invalid mode selected. Running MDP simulation by default.")
        mdp_agent, mdp_rewards, mdp_metrics = run_mdp_simulation(
            env, config, num_episodes=config['simulation']['num_episodes']
        )
    
    # Display final results
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    
    if mdp_rewards:
        print(f"\nMDP Agent Results:")
        print(f"  Average Reward: {np.mean(mdp_rewards):.2f}")
        print(f"  Max Reward: {np.max(mdp_rewards):.2f}")
        print(f"  Final 10 Episodes Avg: {np.mean(mdp_rewards[-10:]):.2f}")
    
    if rl_rewards:
        print(f"\nRL Agent Results:")
        print(f"  Average Reward: {np.mean(rl_rewards):.2f}")
        print(f"  Max Reward: {np.max(rl_rewards):.2f}")
        print(f"  Final 100 Episodes Avg: {np.mean(rl_rewards[-100:]):.2f}")
    
    # Visualize results
    visualize_results(mdp_rewards, rl_rewards, mdp_metrics, rl_metrics, env, config)
    
    # Save models
    if mdp_agent and rl_agent:
        save_models(mdp_agent, rl_agent, config)
    
    print("\n" + "="*80)
    print("Thank you for using the RL Scheduling Simulator!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()