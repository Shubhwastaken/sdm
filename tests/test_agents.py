import unittest
from src.agents.mdp_agent import MDPAgent
from src.agents.rl_agent import RLAgent
from src.environment.scheduling_env import SchedulingEnvironment

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.env = SchedulingEnvironment()
        self.mdp_agent = MDPAgent(self.env)
        self.rl_agent = RLAgent(self.env)

    def test_mdp_agent_initialization(self):
        self.assertIsNotNone(self.mdp_agent)
        self.assertEqual(self.mdp_agent.environment, self.env)

    def test_rl_agent_initialization(self):
        self.assertIsNotNone(self.rl_agent)
        self.assertEqual(self.rl_agent.environment, self.env)

    def test_mdp_agent_action_selection(self):
        state = self.env.reset()
        action = self.mdp_agent.select_action(state)
        self.assertIn(action, self.env.valid_actions)

    def test_rl_agent_action_selection(self):
        state = self.env.reset()
        action = self.rl_agent.select_action(state)
        self.assertIn(action, self.env.valid_actions)

    def test_mdp_agent_learning(self):
        initial_state = self.env.reset()
        action = self.mdp_agent.select_action(initial_state)
        next_state, reward, done, _ = self.env.step(action)
        self.mdp_agent.learn(initial_state, action, reward, next_state)
        # Check if the Q-values are updated (this is a placeholder check)
        self.assertTrue(hasattr(self.mdp_agent, 'q_values'))

    def test_rl_agent_learning(self):
        initial_state = self.env.reset()
        action = self.rl_agent.select_action(initial_state)
        next_state, reward, done, _ = self.env.step(action)
        self.rl_agent.learn(initial_state, action, reward, next_state)
        # Check if the policy is updated (this is a placeholder check)
        self.assertTrue(hasattr(self.rl_agent, 'policy'))

if __name__ == '__main__':
    unittest.main()