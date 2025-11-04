import numpy as np


class StateRepresentation:
    """
    Converts the scheduling environment state into a numerical representation
    suitable for machine learning algorithms.
    """
    
    def __init__(self, num_classes, num_teachers, num_rooms, num_time_slots):
        self.num_classes = num_classes
        self.num_teachers = num_teachers
        self.num_rooms = num_rooms
        self.num_time_slots = num_time_slots
        
        # Calculate state dimension
        self.state_dim = self._calculate_state_dimension()
    
    def _calculate_state_dimension(self):
        """Calculate the total dimension of the state vector"""
        # Schedule matrix: num_classes * 5 (class_id, teacher_id, room_id, time_slot, status)
        schedule_dim = self.num_classes * 5
        
        # Teacher availability: num_teachers * num_time_slots
        teacher_avail_dim = self.num_teachers * self.num_time_slots
        
        # Room availability: num_rooms * num_time_slots
        room_avail_dim = self.num_rooms * self.num_time_slots
        
        # Student enrollment: num_classes
        enrollment_dim = self.num_classes
        
        # Disruption features: count + severity
        disruption_dim = 5  # total_disruptions, teacher_disruptions, room_disruptions, enrollment_disruptions, avg_severity
        
        return schedule_dim + teacher_avail_dim + room_avail_dim + enrollment_dim + disruption_dim
    
    def encode_state(self, schedule, teacher_availability, room_availability, 
                     student_enrollment, active_disruptions):
        """
        Encode the environment state into a flat numerical vector.
        
        Args:
            schedule: numpy array of shape (num_classes, 5)
            teacher_availability: numpy array of shape (num_teachers, num_time_slots)
            room_availability: numpy array of shape (num_rooms, num_time_slots)
            student_enrollment: numpy array of shape (num_classes,)
            active_disruptions: list of disruption dictionaries
        
        Returns:
            state_vector: numpy array of shape (state_dim,)
        """
        state_vector = []
        
        # 1. Schedule information (flattened and normalized)
        schedule_flat = schedule.flatten().astype(np.float32)
        # Normalize IDs and status
        schedule_normalized = self._normalize_schedule(schedule_flat)
        state_vector.extend(schedule_normalized)
        
        # 2. Teacher availability (flattened)
        teacher_avail_flat = teacher_availability.flatten().astype(np.float32)
        state_vector.extend(teacher_avail_flat)
        
        # 3. Room availability (flattened)
        room_avail_flat = room_availability.flatten().astype(np.float32)
        state_vector.extend(room_avail_flat)
        
        # 4. Student enrollment (normalized)
        enrollment_normalized = student_enrollment.astype(np.float32) / 50.0  # Normalize to [0, 1] assuming max 50 students
        state_vector.extend(enrollment_normalized)
        
        # 5. Disruption features
        disruption_features = self._extract_disruption_features(active_disruptions)
        state_vector.extend(disruption_features)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _normalize_schedule(self, schedule_flat):
        """Normalize schedule values to [0, 1] range"""
        normalized = []
        for i, val in enumerate(schedule_flat):
            col = i % 5
            if col == 0:  # class_id
                normalized.append(val / max(1, self.num_classes))
            elif col == 1:  # teacher_id
                normalized.append(val / max(1, self.num_teachers))
            elif col == 2:  # room_id
                normalized.append(val / max(1, self.num_rooms))
            elif col == 3:  # time_slot
                normalized.append(val / max(1, self.num_time_slots))
            elif col == 4:  # status (-1, 0, 1)
                normalized.append((val + 1) / 2.0)  # Map to [0, 1]
        return normalized
    
    def _extract_disruption_features(self, active_disruptions):
        """Extract numerical features from disruptions"""
        if not active_disruptions:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        total_disruptions = len(active_disruptions)
        teacher_disruptions = sum(1 for d in active_disruptions if d['type'] == 'teacher_absence')
        room_disruptions = sum(1 for d in active_disruptions if d['type'] == 'facility_conflict')
        enrollment_disruptions = sum(1 for d in active_disruptions if d['type'] == 'enrollment_change')
        
        # Calculate average severity (low=1, medium=2, high=3)
        severity_map = {'low': 1, 'medium': 2, 'high': 3}
        avg_severity = np.mean([severity_map[d['severity']] for d in active_disruptions]) / 3.0
        
        # Normalize counts
        return [
            min(1.0, total_disruptions / 10.0),
            min(1.0, teacher_disruptions / 5.0),
            min(1.0, room_disruptions / 5.0),
            min(1.0, enrollment_disruptions / 5.0),
            avg_severity
        ]
    
    def get_state_dimension(self):
        """Return the dimension of the state vector"""
        return self.state_dim
    
    def decode_state(self, state_vector):
        """
        Decode a state vector back into human-readable components (for visualization).
        
        Args:
            state_vector: numpy array of shape (state_dim,)
        
        Returns:
            dict: Dictionary containing decoded state components
        """
        idx = 0
        
        # Extract schedule
        schedule_size = self.num_classes * 5
        schedule_flat = state_vector[idx:idx + schedule_size]
        idx += schedule_size
        
        # Extract teacher availability
        teacher_avail_size = self.num_teachers * self.num_time_slots
        teacher_avail = state_vector[idx:idx + teacher_avail_size].reshape(
            (self.num_teachers, self.num_time_slots)
        )
        idx += teacher_avail_size
        
        # Extract room availability
        room_avail_size = self.num_rooms * self.num_time_slots
        room_avail = state_vector[idx:idx + room_avail_size].reshape(
            (self.num_rooms, self.num_time_slots)
        )
        idx += room_avail_size
        
        # Extract enrollment
        enrollment = state_vector[idx:idx + self.num_classes] * 50.0  # Denormalize
        idx += self.num_classes
        
        # Extract disruption features
        disruption_features = state_vector[idx:idx + 5]
        
        return {
            'schedule': schedule_flat.reshape((self.num_classes, 5)),
            'teacher_availability': teacher_avail,
            'room_availability': room_avail,
            'student_enrollment': enrollment,
            'disruption_features': {
                'total': disruption_features[0] * 10,
                'teacher': disruption_features[1] * 5,
                'room': disruption_features[2] * 5,
                'enrollment': disruption_features[3] * 5,
                'avg_severity': disruption_features[4] * 3
            }
        }