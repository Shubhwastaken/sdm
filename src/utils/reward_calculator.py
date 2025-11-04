import numpy as np


class RewardCalculator:
    """
    Calculates rewards for scheduling actions based on multiple criteria:
    - Successful scheduling
    - Resource utilization
    - Handling disruptions
    - Student satisfaction
    """
    
    def __init__(self):
        # Reward weights
        self.weights = {
            'scheduling_success': 10.0,
            'scheduling_failure': -5.0,
            'resource_utilization': 2.0,
            'disruption_handling': -3.0,
            'student_capacity': 1.0,
            'time_efficiency': 1.5,
            'conflict_penalty': -8.0
        }
    
    def calculate_reward(self, schedule, teacher_availability, room_availability, 
                        student_enrollment, success, info):
        """
        Calculate total reward for a scheduling action.
        
        Args:
            schedule: Current schedule state
            teacher_availability: Teacher availability matrix
            room_availability: Room availability matrix
            student_enrollment: Student enrollment per class
            success: Whether the action was successful
            info: Additional information about the action
        
        Returns:
            float: Total reward value
        """
        reward = 0.0
        
        # 1. Basic scheduling success/failure
        if success:
            reward += self.weights['scheduling_success']
        else:
            reason = info.get('reason', '')
            if reason == 'class_already_scheduled':
                reward += self.weights['scheduling_failure'] * 0.5  # Less penalty
            elif reason in ['teacher_unavailable', 'room_unavailable']:
                reward += self.weights['scheduling_failure']
            else:
                reward += self.weights['conflict_penalty']
        
        # 2. Resource utilization efficiency
        teacher_util = 1.0 - np.mean(teacher_availability)
        room_util = 1.0 - np.mean(room_availability)
        avg_utilization = (teacher_util + room_util) / 2.0
        reward += self.weights['resource_utilization'] * avg_utilization
        
        # 3. Progress reward (percentage of classes scheduled)
        num_scheduled = np.sum(schedule[:, 4] == 1)
        total_classes = len(schedule)
        progress = num_scheduled / total_classes
        reward += self.weights['time_efficiency'] * progress
        
        # 4. Penalty for disrupted classes
        num_disrupted = np.sum(schedule[:, 4] == -1)
        if num_disrupted > 0:
            reward += self.weights['disruption_handling'] * num_disrupted
        
        # 5. Student capacity bonus (prefer scheduling larger classes)
        if success and 'class_id' in info:
            class_id = info.get('class_id', 0)
            if class_id < len(student_enrollment):
                enrollment = student_enrollment[class_id]
                # Normalize to [0, 1] assuming max 50 students
                enrollment_factor = min(1.0, enrollment / 50.0)
                reward += self.weights['student_capacity'] * enrollment_factor
        
        return reward
    
    def calculate_episode_reward(self, schedule, initial_classes, disruptions_handled):
        """
        Calculate final reward at end of episode.
        
        Args:
            schedule: Final schedule state
            initial_classes: Number of classes to schedule
            disruptions_handled: Number of disruptions successfully handled
        
        Returns:
            float: Episode completion reward
        """
        # Completion bonus
        num_scheduled = np.sum(schedule[:, 4] == 1)
        completion_rate = num_scheduled / initial_classes
        
        # Big bonus for completing all schedules
        if completion_rate >= 1.0:
            completion_bonus = 50.0
        elif completion_rate >= 0.9:
            completion_bonus = 30.0
        elif completion_rate >= 0.75:
            completion_bonus = 15.0
        else:
            completion_bonus = 0.0
        
        # Disruption handling bonus
        disruption_bonus = disruptions_handled * 5.0
        
        # Penalty for unscheduled classes
        num_unscheduled = np.sum(schedule[:, 4] == 0)
        unscheduled_penalty = num_unscheduled * -2.0
        
        total_reward = completion_bonus + disruption_bonus + unscheduled_penalty
        
        return total_reward
    
    def get_reward_breakdown(self, schedule, teacher_availability, room_availability):
        """
        Get detailed breakdown of reward components for analysis.
        
        Returns:
            dict: Dictionary with reward components
        """
        breakdown = {
            'scheduled': np.sum(schedule[:, 4] == 1),
            'disrupted': np.sum(schedule[:, 4] == -1),
            'unscheduled': np.sum(schedule[:, 4] == 0),
            'teacher_utilization': 1.0 - np.mean(teacher_availability),
            'room_utilization': 1.0 - np.mean(room_availability),
            'completion_rate': np.sum(schedule[:, 4] == 1) / len(schedule)
        }
        
        return breakdown