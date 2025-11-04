import unittest
from src.environment.scheduling_env import SchedulingEnvironment
from src.environment.disruption_generator import DisruptionGenerator

class TestSchedulingEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = SchedulingEnvironment()
        self.disruption_generator = DisruptionGenerator()

    def test_initial_state(self):
        initial_state = self.env.reset()
        self.assertIsNotNone(initial_state)

    def test_step_function(self):
        action = self.env.action_space.sample()  # Assuming action_space is defined
        next_state, reward, done, info = self.env.step(action)
        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_disruption_handling(self):
        disruption = self.disruption_generator.generate_disruption()
        self.env.handle_disruption(disruption)
        self.assertIn(disruption, self.env.current_disruptions)

if __name__ == '__main__':
    unittest.main()