"""
Simple example demonstrating how to use the RL Scheduling Simulator
as a library in your own code.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from environment.scheduling_env import SchedulingEnvironment
from agents.rl_agent import RLAgent
from visualization.plotter import SchedulingPlotter


def example_basic_usage():
    """Example 1: Basic usage with RL agent"""
    print("Example 1: Basic RL Agent Usage\n")
    
    # Define configuration
    env_config = {
        'num_classes': 8,
        'num_teachers': 4,
        'num_rooms': 3,
        'disruption_probability': 0.15
    }
    
    agent_config = {
        'discount_factor': 0.99,
        'learning_rate': 0.01,
        'epsilon': 1.0,
        'exploration_strategy': 'epsilon_greedy'
    }
    
    # Create environment and agent
    env = SchedulingEnvironment(env_config)
    agent = RLAgent(env, agent_config)
    
    # Train for a few episodes
    num_episodes = 30
    rewards = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Episode {episode + 1}: Avg Reward = {avg:.2f}")
    
    print(f"\nTraining complete! Final average: {np.mean(rewards[-10:]):.2f}\n")
    
    return env, agent, rewards


def example_custom_scheduling():
    """Example 2: Custom scheduling scenario"""
    print("Example 2: Custom Scheduling Scenario\n")
    
    # Create a scenario with many disruptions
    env_config = {
        'num_classes': 6,
        'num_teachers': 3,
        'num_rooms': 2,
        'disruption_probability': 0.4  # High disruption rate!
    }
    
    agent_config = {
        'discount_factor': 0.95,
        'learning_rate': 0.05,
        'epsilon': 0.3,  # Lower exploration
    }
    
    env = SchedulingEnvironment(env_config)
    agent = RLAgent(env, agent_config)
    
    # Run one episode and track what happens
    print("Running one episode with high disruption rate...\n")
    
    state = env.reset()
    done = False
    step = 0
    disruptions_encountered = 0
    
    while not done and step < 40:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        if info.get('disruptions', 0) > disruptions_encountered:
            print(f"  Step {step + 1}: DISRUPTION! Total: {info['disruptions']}")
            disruptions_encountered = info['disruptions']
        
        if info.get('success', False):
            print(f"  Step {step + 1}: Successfully scheduled class. Total: {info['scheduled_classes']}")
        
        state = next_state
        step += 1
    
    print(f"\nEpisode finished:")
    print(f"  Steps taken: {step}")
    print(f"  Classes scheduled: {info['scheduled_classes']}")
    print(f"  Disruptions: {info['disruptions']}")
    print()
    
    # Show final schedule
    env.render()
    
    return env


def example_comparing_strategies():
    """Example 3: Compare different learning rates"""
    print("Example 3: Comparing Learning Rates\n")
    
    env_config = {
        'num_classes': 5,
        'num_teachers': 3,
        'num_rooms': 2,
        'disruption_probability': 0.2
    }
    
    learning_rates = [0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        print(f"Testing learning rate = {lr}")
        
        agent_config = {
            'discount_factor': 0.99,
            'learning_rate': lr,
            'epsilon': 1.0,
        }
        
        env = SchedulingEnvironment(env_config)
        agent = RLAgent(env, agent_config)
        
        rewards = []
        for episode in range(20):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards[-10:])
        results[lr] = avg_reward
        print(f"  Average reward (last 10 episodes): {avg_reward:.2f}\n")
    
    print("Results:")
    for lr, avg in results.items():
        print(f"  LR={lr}: {avg:.2f}")
    
    best_lr = max(results, key=results.get)
    print(f"\nBest learning rate: {best_lr} with avg reward {results[best_lr]:.2f}\n")
    
    return results


def example_schedule_analysis():
    """Example 4: Analyze a generated schedule"""
    print("Example 4: Schedule Analysis\n")
    
    env_config = {
        'num_classes': 10,
        'num_teachers': 5,
        'num_rooms': 3,
        'disruption_probability': 0.15
    }
    
    env = SchedulingEnvironment(env_config)
    
    # Generate a schedule using random valid actions
    state = env.reset()
    done = False
    actions_taken = []
    
    print("Generating schedule with random valid actions...\n")
    
    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        action = np.random.choice(valid_actions)
        actions_taken.append(action)
        
        state, reward, done, info = env.step(action)
    
    print(f"Schedule generated with {len(actions_taken)} actions\n")
    
    # Analyze the schedule
    schedule = env.schedule
    
    # Count classes per teacher
    teacher_loads = {}
    for i in range(len(schedule)):
        if schedule[i, 4] == 1:  # If scheduled
            teacher_id = int(schedule[i, 1])
            teacher_loads[teacher_id] = teacher_loads.get(teacher_id, 0) + 1
    
    print("Teacher Workload:")
    for teacher_id in sorted(teacher_loads.keys()):
        print(f"  Teacher {teacher_id}: {teacher_loads[teacher_id]} classes")
    
    # Count classes per room
    room_usage = {}
    for i in range(len(schedule)):
        if schedule[i, 4] == 1:
            room_id = int(schedule[i, 2])
            room_usage[room_id] = room_usage.get(room_id, 0) + 1
    
    print("\nRoom Usage:")
    for room_id in sorted(room_usage.keys()):
        print(f"  Room {room_id}: {room_usage[room_id]} classes")
    
    # Calculate utilization
    total_scheduled = np.sum(schedule[:, 4] == 1)
    total_classes = len(schedule)
    utilization = (total_scheduled / total_classes) * 100
    
    print(f"\nOverall Utilization: {utilization:.1f}% ({total_scheduled}/{total_classes} classes scheduled)")
    print()
    
    return env


def example_visualization():
    """Example 5: Create visualizations"""
    print("Example 5: Visualization\n")
    
    # Train a quick agent
    env_config = {'num_classes': 8, 'num_teachers': 4, 'num_rooms': 3, 'disruption_probability': 0.2}
    agent_config = {'discount_factor': 0.99, 'learning_rate': 0.01, 'epsilon': 1.0}
    
    env = SchedulingEnvironment(env_config)
    agent = RLAgent(env, agent_config)
    
    print("Training agent...")
    rewards = agent.train(num_episodes=50, verbose=False)
    
    print(f"Training complete! Average reward: {np.mean(rewards[-10:]):.2f}\n")
    
    # Create visualizations
    print("Creating visualizations...")
    plotter = SchedulingPlotter()
    
    os.makedirs('example_output', exist_ok=True)
    
    plotter.plot_training_progress(rewards, save_path='example_output/training.png')
    print("  ✓ Training progress plot saved")
    
    # Generate a final schedule
    state = env.reset()
    done = False
    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
    
    plotter.plot_schedule_heatmap(
        env.schedule, 
        env_config['num_teachers'], 
        env_config['num_rooms'], 
        env.num_time_slots,
        save_path='example_output/heatmap.png'
    )
    print("  ✓ Schedule heatmap saved")
    
    if env.active_disruptions:
        plotter.plot_disruption_analysis(env.active_disruptions, save_path='example_output/disruptions.png')
        print("  ✓ Disruption analysis saved")
    
    print("\nVisualizations saved to 'example_output/' directory\n")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" RL SCHEDULING SIMULATOR - CODE EXAMPLES")
    print("="*80 + "\n")
    
    print("This script demonstrates how to use the simulator in your own code.\n")
    
    # Run examples
    try:
        # Example 1
        env, agent, rewards = example_basic_usage()
        
        # Example 2
        env2 = example_custom_scheduling()
        
        # Example 3
        results = example_comparing_strategies()
        
        # Example 4
        env4 = example_schedule_analysis()
        
        # Example 5
        example_visualization()
        
        print("="*80)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nYou can now:")
        print("  • Modify these examples for your needs")
        print("  • Import the modules in your own scripts")
        print("  • Build custom scheduling scenarios")
        print("  • Experiment with different configurations")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
