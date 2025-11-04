import numpy as np
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.disruption_generator import DisruptionGenerator
from utils.reward_calculator import RewardCalculator
from utils.state_representation import StateRepresentation


class SchedulingEnvironment:
    """
    Scheduling Environment for RL-based schedule optimization.
    Handles teacher assignments, room allocations, and disruptions.
    """
    
    def __init__(self, config):
        self.config = config
        self.num_classes = config.get('num_classes', 10)
        self.num_teachers = config.get('num_teachers', 5)
        self.num_rooms = config.get('num_rooms', 3)
        self.max_disruptions = config.get('max_disruptions', 5)
        self.disruption_probability = config.get('disruption_probability', 0.2)
        
        # Time slots (e.g., 8 slots per day)
        self.num_time_slots = 8
        
        # Initialize components
        self.disruption_gen = DisruptionGenerator(
            self.num_teachers, 
            self.num_rooms, 
            self.disruption_probability,
            self.num_classes  # Pass num_classes
        )
        self.reward_calculator = RewardCalculator()
        self.state_repr = StateRepresentation(
            self.num_classes, 
            self.num_teachers, 
            self.num_rooms, 
            self.num_time_slots
        )
        
        # Current state variables
        self.current_step = 0
        self.max_steps = 50
        self.schedule = None
        self.teacher_availability = None
        self.room_availability = None
        self.student_enrollment = None
        self.active_disruptions = []
        
        # Action space: (class_id, teacher_id, room_id, time_slot)
        self.action_space_size = self.num_classes * self.num_teachers * self.num_rooms * self.num_time_slots
        
        # State space dimension
        self.state_space_size = self.state_repr.get_state_dimension()
        
        self.state = self.initialize_state()
        self.done = False

    def initialize_state(self):
        """Initialize the scheduling state with default values"""
        # Schedule: [class_id, teacher_id, room_id, time_slot, status]
        # status: 0 = not scheduled, 1 = scheduled, -1 = disrupted
        self.schedule = np.zeros((self.num_classes, 5), dtype=np.int32)
        self.schedule[:, 0] = np.arange(self.num_classes)  # class IDs
        
        # Teacher availability: [teacher_id][time_slot] = 1 if available
        self.teacher_availability = np.ones((self.num_teachers, self.num_time_slots), dtype=np.int32)
        
        # Room availability: [room_id][time_slot] = 1 if available
        self.room_availability = np.ones((self.num_rooms, self.num_time_slots), dtype=np.int32)
        
        # Student enrollment per class (randomized for realism)
        self.student_enrollment = np.random.randint(15, 40, size=self.num_classes)
        
        # Active disruptions list
        self.active_disruptions = []
        
        # Generate initial state representation
        state = self.state_repr.encode_state(
            self.schedule,
            self.teacher_availability,
            self.room_availability,
            self.student_enrollment,
            self.active_disruptions
        )
        
        return state

    def step(self, action):
        """
        Execute one step in the environment.
        Action format: integer representing (class_id, teacher_id, room_id, time_slot)
        """
        self.current_step += 1
        
        # Decode action
        class_id, teacher_id, room_id, time_slot = self._decode_action(action)
        
        # Apply action
        success, info = self._apply_action(class_id, teacher_id, room_id, time_slot)
        
        # Generate random disruptions
        if np.random.random() < self.disruption_probability:
            disruption = self.disruption_gen.generate_disruption()
            self._apply_disruption(disruption)
            self.active_disruptions.append(disruption)
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self.schedule,
            self.teacher_availability,
            self.room_availability,
            self.student_enrollment,
            success,
            info
        )
        
        # Get new state
        new_state = self.state_repr.encode_state(
            self.schedule,
            self.teacher_availability,
            self.room_availability,
            self.student_enrollment,
            self.active_disruptions
        )
        
        # Check if done
        self.done = self._check_done()
        
        # Additional info
        info_dict = {
            'success': success,
            'disruptions': len(self.active_disruptions),
            'scheduled_classes': np.sum(self.schedule[:, 4] == 1),
            'step': self.current_step,
            **info
        }
        
        self.state = new_state
        return new_state, reward, self.done, info_dict

    def _decode_action(self, action):
        """Decode integer action to schedule parameters"""
        # Simple encoding: action = class_id * (T*R*S) + teacher_id * (R*S) + room_id * S + time_slot
        divisor = self.num_teachers * self.num_rooms * self.num_time_slots
        class_id = min(action // divisor, self.num_classes - 1)
        remainder = action % divisor
        
        divisor = self.num_rooms * self.num_time_slots
        teacher_id = min(remainder // divisor, self.num_teachers - 1)
        remainder = remainder % divisor
        
        room_id = min(remainder // self.num_time_slots, self.num_rooms - 1)
        time_slot = min(remainder % self.num_time_slots, self.num_time_slots - 1)
        
        return class_id, teacher_id, room_id, time_slot

    def _apply_action(self, class_id, teacher_id, room_id, time_slot):
        """Apply scheduling action and return success status"""
        info = {}
        
        # Check if class is already scheduled
        if self.schedule[class_id, 4] == 1:
            info['reason'] = 'class_already_scheduled'
            return False, info
        
        # Check teacher availability
        if self.teacher_availability[teacher_id, time_slot] == 0:
            info['reason'] = 'teacher_unavailable'
            return False, info
        
        # Check room availability
        if self.room_availability[room_id, time_slot] == 0:
            info['reason'] = 'room_unavailable'
            return False, info
        
        # Apply the schedule
        self.schedule[class_id, 1] = teacher_id
        self.schedule[class_id, 2] = room_id
        self.schedule[class_id, 3] = time_slot
        self.schedule[class_id, 4] = 1  # status = scheduled
        
        # Update availability
        self.teacher_availability[teacher_id, time_slot] = 0
        self.room_availability[room_id, time_slot] = 0
        
        info['reason'] = 'success'
        return True, info

    def _apply_disruption(self, disruption):
        """Apply a disruption to the current schedule"""
        disruption_type = disruption['type']
        
        if disruption_type == 'teacher_absence':
            teacher_id = disruption['teacher_id']
            # Mark teacher as unavailable for all time slots
            self.teacher_availability[teacher_id, :] = 0
            
            # Mark affected classes as disrupted
            for i in range(self.num_classes):
                if self.schedule[i, 1] == teacher_id and self.schedule[i, 4] == 1:
                    self.schedule[i, 4] = -1  # disrupted status
                    
        elif disruption_type == 'facility_conflict':
            room_id = disruption['room_id']
            time_slot = disruption['time_slot']
            # Mark room as unavailable for specific time slot
            self.room_availability[room_id, time_slot] = 0
            
            # Mark affected class as disrupted
            for i in range(self.num_classes):
                if (self.schedule[i, 2] == room_id and 
                    self.schedule[i, 3] == time_slot and 
                    self.schedule[i, 4] == 1):
                    self.schedule[i, 4] = -1
                    
        elif disruption_type == 'enrollment_change':
            class_id = disruption['class_id']
            change = disruption['change']
            self.student_enrollment[class_id] = max(0, self.student_enrollment[class_id] + change)

    def _check_done(self):
        """Check if episode is done"""
        # Done if all classes scheduled or max steps reached
        all_scheduled = np.all(self.schedule[:, 4] >= 1)
        max_steps_reached = self.current_step >= self.max_steps
        
        return all_scheduled or max_steps_reached

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.done = False
        self.active_disruptions = []
        self.state = self.initialize_state()
        return self.state

    def render(self, mode='human'):
        """Render the current schedule"""
        print("\n" + "="*80)
        print(f"SCHEDULE STATUS - Step {self.current_step}")
        print("="*80)
        
        print(f"\nScheduled Classes: {np.sum(self.schedule[:, 4] == 1)}/{self.num_classes}")
        print(f"Disrupted Classes: {np.sum(self.schedule[:, 4] == -1)}")
        print(f"Active Disruptions: {len(self.active_disruptions)}")
        
        print("\nClass Schedule:")
        print(f"{'Class':<8} {'Teacher':<10} {'Room':<8} {'Time':<8} {'Status':<12} {'Students':<10}")
        print("-" * 70)
        
        for i in range(self.num_classes):
            class_id = self.schedule[i, 0]
            teacher_id = self.schedule[i, 1]
            room_id = self.schedule[i, 2]
            time_slot = self.schedule[i, 3]
            status = self.schedule[i, 4]
            students = self.student_enrollment[i]
            
            status_str = {-1: 'DISRUPTED', 0: 'NOT SCHEDULED', 1: 'SCHEDULED'}[status]
            
            print(f"{class_id:<8} {teacher_id:<10} {room_id:<8} {time_slot:<8} {status_str:<12} {students:<10}")
        
        if self.active_disruptions:
            print("\nActive Disruptions:")
            for disruption in self.active_disruptions[-3:]:  # Show last 3
                print(f"  - {disruption}")
        
        print("="*80 + "\n")

    def get_valid_actions(self):
        """Return list of valid actions in current state"""
        valid_actions = []
        
        for class_id in range(self.num_classes):
            # Only consider unscheduled or disrupted classes
            if self.schedule[class_id, 4] <= 0:
                for teacher_id in range(self.num_teachers):
                    for room_id in range(self.num_rooms):
                        for time_slot in range(self.num_time_slots):
                            # Check availability
                            if (self.teacher_availability[teacher_id, time_slot] == 1 and
                                self.room_availability[room_id, time_slot] == 1):
                                action = self._encode_action(class_id, teacher_id, room_id, time_slot)
                                valid_actions.append(action)
        
        return valid_actions

    def _encode_action(self, class_id, teacher_id, room_id, time_slot):
        """Encode schedule parameters to integer action"""
        action = (class_id * self.num_teachers * self.num_rooms * self.num_time_slots +
                  teacher_id * self.num_rooms * self.num_time_slots +
                  room_id * self.num_time_slots +
                  time_slot)
        return action