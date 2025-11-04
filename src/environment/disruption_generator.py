import numpy as np
import random


class DisruptionGenerator:
    """
    Generates various types of disruptions in the scheduling system:
    - Teacher absences
    - Facility conflicts
    - Student enrollment changes
    """
    
    def __init__(self, num_teachers, num_rooms, disruption_probability=0.2, num_classes=10):
        self.num_teachers = num_teachers
        self.num_rooms = num_rooms
        self.num_classes = num_classes
        self.disruption_probability = disruption_probability
        
        # Disruption types and their probabilities
        self.disruption_types = {
            'teacher_absence': 0.4,
            'facility_conflict': 0.35,
            'enrollment_change': 0.25
        }
        
        # Severity levels
        self.severity_levels = ['low', 'medium', 'high']
        self.severity_weights = [0.5, 0.35, 0.15]
    
    def generate_disruption(self):
        """Generate a random disruption event"""
        disruption_type = random.choices(
            list(self.disruption_types.keys()),
            weights=list(self.disruption_types.values())
        )[0]
        
        severity = random.choices(
            self.severity_levels,
            weights=self.severity_weights
        )[0]
        
        disruption = {
            'type': disruption_type,
            'severity': severity,
            'timestamp': None  # Can be set by environment
        }
        
        if disruption_type == 'teacher_absence':
            disruption.update(self._generate_teacher_absence(severity))
        elif disruption_type == 'facility_conflict':
            disruption.update(self._generate_facility_conflict(severity))
        elif disruption_type == 'enrollment_change':
            disruption.update(self._generate_enrollment_change(severity))
        
        return disruption
    
    def _generate_teacher_absence(self, severity):
        """Generate a teacher absence disruption"""
        teacher_id = np.random.randint(0, self.num_teachers)
        
        # Duration based on severity
        duration_map = {
            'low': 1,      # 1 time slot
            'medium': 4,   # half day
            'high': 8      # full day
        }
        duration = duration_map[severity]
        
        return {
            'teacher_id': teacher_id,
            'duration': duration,
            'reason': random.choice(['sick', 'emergency', 'meeting', 'training'])
        }
    
    def _generate_facility_conflict(self, severity):
        """Generate a facility/room conflict disruption"""
        room_id = np.random.randint(0, self.num_rooms)
        time_slot = np.random.randint(0, 8)  # Assuming 8 time slots
        
        # Duration based on severity
        duration_map = {
            'low': 1,
            'medium': 2,
            'high': 4
        }
        duration = duration_map[severity]
        
        return {
            'room_id': room_id,
            'time_slot': time_slot,
            'duration': duration,
            'reason': random.choice(['maintenance', 'double_booking', 'equipment_failure', 'renovation'])
        }
    
    def _generate_enrollment_change(self, severity):
        """Generate a student enrollment change disruption"""
        class_id = np.random.randint(0, self.num_classes)  # Use instance variable
        
        # Change magnitude based on severity
        change_map = {
            'low': np.random.randint(-3, 4),
            'medium': np.random.randint(-8, 9),
            'high': np.random.randint(-15, 16)
        }
        change = change_map[severity]
        
        return {
            'class_id': class_id,
            'change': change,
            'reason': random.choice(['add_drop', 'transfer', 'new_enrollment', 'withdrawal'])
        }
    
    def generate_disruption_sequence(self, num_disruptions):
        """Generate a sequence of disruptions for testing"""
        return [self.generate_disruption() for _ in range(num_disruptions)]
    
    def get_disruption_severity_score(self, disruption):
        """Calculate a severity score for a disruption (0-10)"""
        severity_scores = {'low': 3, 'medium': 6, 'high': 9}
        base_score = severity_scores[disruption['severity']]
        
        # Adjust based on disruption type
        type_multipliers = {
            'teacher_absence': 1.2,
            'facility_conflict': 1.0,
            'enrollment_change': 0.8
        }
        multiplier = type_multipliers[disruption['type']]
        
        return min(10, base_score * multiplier)