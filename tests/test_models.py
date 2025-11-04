import unittest
from src.models.q_learning import QLearning
from src.models.policy_network import PolicyNetwork

class TestModels(unittest.TestCase):

    def setUp(self):
        self.q_learning_model = QLearning(state_size=10, action_size=5)
        self.policy_network_model = PolicyNetwork(input_size=10, output_size=5)

    def test_q_learning_initialization(self):
        self.assertEqual(self.q_learning_model.state_size, 10)
        self.assertEqual(self.q_learning_model.action_size, 5)
        self.assertIsNotNone(self.q_learning_model.q_table)

    def test_policy_network_initialization(self):
        self.assertEqual(self.policy_network_model.input_size, 10)
        self.assertEqual(self.policy_network_model.output_size, 5)
        self.assertIsNotNone(self.policy_network_model.model)

    def test_q_learning_update(self):
        initial_value = self.q_learning_model.q_table[0][0]
        self.q_learning_model.update_q_value(state=0, action=0, reward=1, next_state=1)
        updated_value = self.q_learning_model.q_table[0][0]
        self.assertNotEqual(initial_value, updated_value)

    def test_policy_network_forward_pass(self):
        import numpy as np
        test_input = np.random.rand(1, 10)
        output = self.policy_network_model.forward(test_input)
        self.assertEqual(output.shape, (1, 5))

if __name__ == '__main__':
    unittest.main()