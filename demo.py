"""
Quick demonstration of the RL Scheduling Simulator
Run this file to see a quick demo of the system in action.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
import numpy as np
from environment.scheduling_env import SchedulingEnvironment
from agents.mdp_agent import MDPAgent
from agents.rl_agent import RLAgent
from visualization.plotter import SchedulingPlotter


def demo_environment():
    """Demonstrate the environment functionality"""
    print("\n" + "="*80)
    print("DEMO 1: ENVIRONMENT BASICS")
    print("="*80 + "\n")
    
    # Create a simple environment
    config = {
        'num_classes': 5,
        'num_teachers': 3,
        'num_rooms': 2,
        'max_disruptions': 3,
        'disruption_probability': 0.1
    }
    
    env = SchedulingEnvironment(config)
    
    print("Environment created with:")
    print(f"  - {config['num_classes']} classes")
    print(f"  - {config['num_teachers']} teachers")
    print(f"  - {config['num_rooms']} rooms")
    print(f"  - {env.num_time_slots} time slots")
    
    # Reset and take a few random actions
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"State space size: {env.state_space_size}")
    print(f"Action space size: {env.action_space_size}")
    
    print("\nTaking 10 random valid actions...")
    for i in range(10):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        action = np.random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: Reward={reward:.2f}, Scheduled={info['scheduled_classes']}, Done={done}")
        
        if done:
            break
    
    print("\nFinal schedule state:")
    env.render()
    
    return env


def demo_mdp_agent():
    """Demonstrate MDP agent"""
    print("\n" + "="*80)
    print("DEMO 2: MDP AGENT")
    print("="*80 + "\n")
    
    # Create environment
    config = {
        'num_classes': 5,
        'num_teachers': 3,
        'num_rooms': 2,
        'max_disruptions': 2,
        'disruption_probability': 0.1
    }
    
    env = SchedulingEnvironment(config)
    
    # Create MDP agent
    agent_config = {
        'discount_factor': 0.9,
        'learning_rate': 0.1,
        'exploration_rate': 0.1
    }
    
    agent = MDPAgent(env, agent_config)
    
    print("Training MDP agent for 20 episodes...")
    rewards = []
    
    for episode in range(20):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 30:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        rewards.append(episode_reward)
        
        if (episode + 1) % 5 == 0:
            agent.value_iteration(max_iterations=10)
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Scheduled={info['scheduled_classes']}")
    
    print(f"\nMDP Agent Training Complete!")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Final 5 Episodes Avg: {np.mean(rewards[-5:]):.2f}")
    
    return agent, rewards


def demo_rl_agent():
    """Demonstrate RL agent"""
    print("\n" + "="*80)
    print("DEMO 3: RL AGENT (Q-LEARNING)")
    print("="*80 + "\n")
    
    # Create environment
    config = {
        'num_classes': 5,
        'num_teachers': 3,
        'num_rooms': 2,
        'max_disruptions': 2,
        'disruption_probability': 0.1
    }
    
    env = SchedulingEnvironment(config)
    
    # Create RL agent
    agent_config = {
        'discount_factor': 0.99,
        'learning_rate': 0.01,
        'epsilon': 1.0,
        'exploration_strategy': 'epsilon_greedy'
    }
    
    agent = RLAgent(env, agent_config)
    
    print("Training RL agent for 50 episodes...")
    rewards = []
    
    for episode in range(50):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 30:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        rewards.append(episode_reward)
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Epsilon={agent.epsilon:.3f}, Scheduled={info['scheduled_classes']}")
    
    print(f"\nRL Agent Training Complete!")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Final 10 Episodes Avg: {np.mean(rewards[-10:]):.2f}")
    
    return agent, rewards


def demo_disruptions():
    """Demonstrate disruption handling"""
    print("\n" + "="*80)
    print("DEMO 4: DISRUPTION HANDLING")
    print("="*80 + "\n")
    
    from environment.disruption_generator import DisruptionGenerator
    
    # Create disruption generator
    gen = DisruptionGenerator(num_teachers=5, num_rooms=3, disruption_probability=0.3)
    
    print("Generating 10 random disruptions:\n")
    
    disruptions = []
    for i in range(10):
        disruption = gen.generate_disruption()
        disruptions.append(disruption)
        
        severity_score = gen.get_disruption_severity_score(disruption)
        
        print(f"{i+1}. Type: {disruption['type']:<20} "
              f"Severity: {disruption['severity']:<8} "
              f"Score: {severity_score:.1f}/10")
        
        # Print specific details
        if disruption['type'] == 'teacher_absence':
            print(f"   → Teacher {disruption['teacher_id']}, "
                  f"Duration: {disruption['duration']} slots, "
                  f"Reason: {disruption['reason']}")
        elif disruption['type'] == 'facility_conflict':
            print(f"   → Room {disruption['room_id']}, "
                  f"Time slot: {disruption['time_slot']}, "
                  f"Reason: {disruption['reason']}")
        elif disruption['type'] == 'enrollment_change':
            print(f"   → Class {disruption['class_id']}, "
                  f"Change: {disruption['change']:+d} students, "
                  f"Reason: {disruption['reason']}")
    
    return disruptions


def demo_visualization(mdp_rewards, rl_rewards):
    """Demonstrate visualization"""
    print("\n" + "="*80)
    print("DEMO 5: VISUALIZATION")
    print("="*80 + "\n")
    
    plotter = SchedulingPlotter()
    
    # Create output directory
    os.makedirs('demo_output', exist_ok=True)
    
    print("Generating comparison plot...")
    plotter.plot_comparison(mdp_rewards, rl_rewards, save_path='demo_output/comparison.png')
    
    print("Visualization saved to 'demo_output/comparison.png'")
    print("\nNote: Close the plot window to continue...")


def main():
    """Run all demos"""
    print("\n")
    print("="*80)
    print(" RL SCHEDULING SIMULATOR - QUICK DEMO")
    print("="*80)
    print("\nThis demo will showcase the key features of the simulator.")
    print("Each demo will run automatically. Press Ctrl+C to exit.\n")
    
    try:
        # Demo 1: Environment
        env = demo_environment()
        
        # Demo 2: MDP Agent
        mdp_agent, mdp_rewards = demo_mdp_agent()
        
        # Demo 3: RL Agent
        rl_agent, rl_rewards = demo_rl_agent()
        
        # Demo 4: Disruptions
        disruptions = demo_disruptions()
        
        # Demo 5: Visualization
        demo_visualization(mdp_rewards, rl_rewards)
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("\nYou've seen:")
        print("  ✓ Environment creation and interaction")
        print("  ✓ MDP agent training")
        print("  ✓ RL agent training")
        print("  ✓ Disruption generation and handling")
        print("  ✓ Performance visualization")
        print("\nTo run the full simulator, execute: python src/main.py")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
