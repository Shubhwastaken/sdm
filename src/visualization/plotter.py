import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class SchedulingPlotter:
    """
    Visualization tools for the scheduling simulation.
    """
    
    def __init__(self, style='seaborn-v0_8'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_progress(self, episode_rewards, window_size=100, save_path=None):
        """
        Plot training progress with rewards over episodes.
        
        Args:
            episode_rewards: List of rewards per episode
            window_size: Moving average window size
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Raw rewards
        ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
        
        # Moving average
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax1.plot(range(window_size-1, len(episode_rewards)), 
                    moving_avg, 
                    color='red', 
                    linewidth=2, 
                    label=f'{window_size}-Episode Moving Average')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Progress: Rewards over Episodes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative rewards
        cumulative_rewards = np.cumsum(episode_rewards)
        ax2.plot(cumulative_rewards, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Cumulative Rewards')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_schedule_heatmap(self, schedule, num_teachers, num_rooms, num_time_slots, save_path=None):
        """
        Plot schedule as a heatmap showing resource allocation.
        
        Args:
            schedule: Schedule array (num_classes x 5)
            num_teachers: Number of teachers
            num_rooms: Number of rooms
            num_time_slots: Number of time slots
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Teacher allocation heatmap
        teacher_schedule = np.zeros((num_teachers, num_time_slots))
        for i in range(len(schedule)):
            if schedule[i, 4] == 1:  # If scheduled
                teacher_id = int(schedule[i, 1])
                time_slot = int(schedule[i, 3])
                teacher_schedule[teacher_id, time_slot] += 1
        
        sns.heatmap(teacher_schedule, annot=True, fmt='g', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': 'Classes'})
        ax1.set_xlabel('Time Slot')
        ax1.set_ylabel('Teacher ID')
        ax1.set_title('Teacher Allocation Across Time Slots')
        
        # Room allocation heatmap
        room_schedule = np.zeros((num_rooms, num_time_slots))
        for i in range(len(schedule)):
            if schedule[i, 4] == 1:  # If scheduled
                room_id = int(schedule[i, 2])
                time_slot = int(schedule[i, 3])
                room_schedule[room_id, time_slot] += 1
        
        sns.heatmap(room_schedule, annot=True, fmt='g', cmap='YlGnBu', 
                   ax=ax2, cbar_kws={'label': 'Classes'})
        ax2.set_xlabel('Time Slot')
        ax2.set_ylabel('Room ID')
        ax2.set_title('Room Allocation Across Time Slots')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_disruption_analysis(self, disruptions, save_path=None):
        """
        Analyze and visualize disruption patterns.
        
        Args:
            disruptions: List of disruption dictionaries
            save_path: Path to save the figure
        """
        if not disruptions:
            print("No disruptions to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Disruption types distribution
        disruption_types = [d['type'] for d in disruptions]
        type_counts = {}
        for dtype in disruption_types:
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        axes[0, 0].bar(type_counts.keys(), type_counts.values(), color='coral')
        axes[0, 0].set_xlabel('Disruption Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Disruption Types')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Severity distribution
        severities = [d['severity'] for d in disruptions]
        severity_counts = {}
        for sev in severities:
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        axes[0, 1].pie(severity_counts.values(), labels=severity_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Severity Distribution')
        
        # Disruptions over time
        axes[1, 0].plot(range(len(disruptions)), range(len(disruptions)), 
                       marker='o', linestyle='-', color='purple')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Cumulative Disruptions')
        axes[1, 0].set_title('Disruptions Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Severity score timeline
        severity_map = {'low': 1, 'medium': 2, 'high': 3}
        severity_scores = [severity_map[d['severity']] for d in disruptions]
        axes[1, 1].scatter(range(len(severity_scores)), severity_scores, 
                          c=severity_scores, cmap='RdYlGn_r', s=100, alpha=0.6)
        axes[1, 1].set_xlabel('Disruption Number')
        axes[1, 1].set_ylabel('Severity Score')
        axes[1, 1].set_title('Severity Scores Timeline')
        axes[1, 1].set_yticks([1, 2, 3])
        axes[1, 1].set_yticklabels(['Low', 'Medium', 'High'])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self, mdp_rewards, rl_rewards, save_path=None):
        """
        Compare performance of MDP and RL agents.
        
        Args:
            mdp_rewards: List of MDP agent rewards
            rl_rewards: List of RL agent rewards
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Episode rewards comparison
        min_len = min(len(mdp_rewards), len(rl_rewards))
        episodes = range(min_len)
        
        axes[0].plot(episodes, mdp_rewards[:min_len], 
                    label='MDP Agent', color='blue', alpha=0.7, linewidth=2)
        axes[0].plot(episodes, rl_rewards[:min_len], 
                    label='RL Agent', color='red', alpha=0.7, linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Agent Performance Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        data = [mdp_rewards, rl_rewards]
        axes[1].boxplot(data, labels=['MDP Agent', 'RL Agent'])
        axes[1].set_ylabel('Reward')
        axes[1].set_title('Reward Distribution Comparison')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mdp_mean = np.mean(mdp_rewards)
        rl_mean = np.mean(rl_rewards)
        axes[1].text(1, mdp_mean, f'μ={mdp_mean:.2f}', 
                    ha='right', va='bottom', fontsize=10, color='blue')
        axes[1].text(2, rl_mean, f'μ={rl_mean:.2f}', 
                    ha='left', va='bottom', fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_resource_utilization(self, teacher_utilization, room_utilization, 
                                  time_slots, save_path=None):
        """
        Plot resource utilization over time.
        
        Args:
            teacher_utilization: Array of teacher utilization per time slot
            room_utilization: Array of room utilization per time slot
            time_slots: List of time slot labels
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(time_slots))
        width = 0.35
        
        ax.bar(x - width/2, teacher_utilization, width, 
              label='Teacher Utilization', color='steelblue')
        ax.bar(x + width/2, room_utilization, width, 
              label='Room Utilization', color='lightcoral')
        
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Resource Utilization by Time Slot')
        ax.set_xticks(x)
        ax.set_xticklabels(time_slots)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scheduling_metrics(self, metrics_history, save_path=None):
        """
        Plot various scheduling metrics over time.
        
        Args:
            metrics_history: List of dictionaries with metrics per episode
            save_path: Path to save the figure
        """
        if not metrics_history:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        episodes = range(len(metrics_history))
        
        # Scheduled classes
        scheduled = [m.get('scheduled_classes', 0) for m in metrics_history]
        axes[0, 0].plot(episodes, scheduled, color='green', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Classes Scheduled')
        axes[0, 0].set_title('Scheduling Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Disruptions
        disruptions = [m.get('disruptions', 0) for m in metrics_history]
        axes[0, 1].plot(episodes, disruptions, color='red', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Number of Disruptions')
        axes[0, 1].set_title('Disruptions per Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        success = [m.get('success', 0) for m in metrics_history]
        axes[1, 0].plot(episodes, success, color='blue', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Scheduling Success Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Steps per episode
        steps = [m.get('step', 0) for m in metrics_history]
        axes[1, 1].plot(episodes, steps, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps Taken')
        axes[1, 1].set_title('Episode Length')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_agent_performance(episode_rewards, title='Agent Performance'):
    """Legacy function for backward compatibility"""
    plotter = SchedulingPlotter()
    plotter.plot_training_progress(episode_rewards)


def plot_scheduling_efficiency(efficiency_data, title='Scheduling Efficiency'):
    """Legacy function for backward compatibility"""
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(efficiency_data)), efficiency_data, color='green')
    plt.xlabel('Scheduling Instance')
    plt.ylabel('Efficiency')
    plt.title(title)
    plt.xticks(range(len(efficiency_data)), [f'Instance {i+1}' for i in range(len(efficiency_data))])
    plt.grid(axis='y')
    plt.show()


def plot_disruption_impact(disruption_data, title='Impact of Disruptions'):
    """Legacy function for backward compatibility"""
    plt.figure(figsize=(10, 5))
    plt.plot(disruption_data, label='Disruption Impact', color='red')
    plt.xlabel('Time')
    plt.ylabel('Impact Level')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()