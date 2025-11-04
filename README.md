# RL-Based Scheduling Simulator

A sophisticated **Reinforcement Learning (RL) and Markov Decision Process (MDP)** based scheduling system that adapts to unpredictable disruptions in real-time. This simulator handles teacher absences, facility conflicts, and student enrollment fluctuations using intelligent optimization algorithms.

## 🌟 Features

- **Multiple Agent Types:**
  - **MDP Agent**: Uses value iteration and policy optimization
  - **RL Agent**: Q-Learning with experience replay (supports Deep Q-Network)
  - **Hybrid Mode**: Combines MDP and RL for optimal performance

- **Realistic Disruptions:**
  - Teacher absences (sick leave, emergencies, training)
  - Facility conflicts (maintenance, double-booking, equipment failures)
  - Student enrollment changes (add/drop, transfers, new enrollments)

- **Dynamic State Representation:**
  - Real-time scheduling state encoding
  - Teacher and room availability tracking
  - Student enrollment monitoring
  - Disruption severity analysis

- **Comprehensive Reward System:**
  - Successful scheduling rewards
  - Resource utilization optimization
  - Disruption handling penalties
  - Student capacity considerations

- **Rich Visualizations:**
  - Training progress plots
  - Schedule heatmaps
  - Disruption analysis
  - Agent performance comparison
  - Resource utilization charts

## 📋 Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- PyYAML

Optional (for Deep Q-Network):
- TensorFlow 2.8+ or PyTorch 1.10+

## 🚀 Installation

1. Clone the repository:
```bash
cd rl-scheduling-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install TensorFlow for Deep Q-Network:
```bash
pip install tensorflow>=2.8.0
```

## 📖 Usage

### Basic Usage

Run the simulator with interactive mode selection:

```bash
cd src
python main.py
```

You'll be prompted to select a simulation mode:
1. **MDP Only** - Pure Markov Decision Process approach
2. **RL Only** - Reinforcement Learning with Q-Learning
3. **Hybrid** - Combined MDP + RL approach
4. **Compare MDP vs RL** - Run both and compare results

### Configuration

Edit `config/simulation_config.yaml` to customize:

```yaml
environment:
  num_classes: 10          # Number of classes to schedule
  num_teachers: 5          # Available teachers
  num_rooms: 3             # Available rooms
  disruption_probability: 0.2  # Probability of disruption per step

agents:
  mdp:
    discount_factor: 0.9
    learning_rate: 0.1
    exploration_rate: 0.1
  
  rl:
    discount_factor: 0.99
    learning_rate: 0.001
    epsilon: 1.0

simulation:
  num_episodes: 200
  max_steps_per_episode: 50
  log_interval: 20
```

### Example: Running MDP Agent

```python
from environment.scheduling_env import SchedulingEnvironment
from agents.mdp_agent import MDPAgent
import yaml

# Load config
with open('../config/simulation_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize environment and agent
env = SchedulingEnvironment(config['environment'])
agent = MDPAgent(env, config['agents']['mdp'])

# Train the agent
rewards = agent.train(num_episodes=100)

# Save the policy
agent.save_policy('../models/mdp_policy.pkl')
```

### Example: Running RL Agent

```python
from environment.scheduling_env import SchedulingEnvironment
from agents.rl_agent import RLAgent
import yaml

# Load config
with open('../config/simulation_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize environment and agent
env = SchedulingEnvironment(config['environment'])
agent = RLAgent(env, config['agents']['rl'])

# Train the agent
rewards = agent.train(num_episodes=1000)

# Save the model
agent.save_model('../models/rl_agent.pkl')
```

## 🏗️ Architecture

### Project Structure

```
rl-scheduling-simulator/
├── src/
│   ├── main.py                      # Main simulation entry point
│   ├── environment/
│   │   ├── scheduling_env.py        # Core scheduling environment
│   │   └── disruption_generator.py  # Disruption simulation
│   ├── agents/
│   │   ├── mdp_agent.py            # MDP-based agent
│   │   └── rl_agent.py             # RL-based agent (Q-Learning/DQN)
│   ├── utils/
│   │   ├── state_representation.py  # State encoding/decoding
│   │   └── reward_calculator.py     # Reward computation
│   └── visualization/
│       └── plotter.py              # Visualization tools
├── config/
│   └── simulation_config.yaml      # Configuration file
├── data/
│   └── sample_schedules.json       # Sample schedule data
├── models/                         # Saved models directory
├── output/                         # Visualization outputs
└── requirements.txt
```

### Environment Design

The `SchedulingEnvironment` implements a gym-like interface:

- **State Space**: Includes schedule matrix, teacher/room availability, student enrollment, and active disruptions
- **Action Space**: Tuple of (class_id, teacher_id, room_id, time_slot)
- **Reward Function**: Multi-objective reward based on scheduling success, resource utilization, and disruption handling

### Agent Algorithms

#### MDP Agent
- **Value Iteration**: Computes optimal value function
- **Policy Extraction**: Derives optimal policy from values
- **Model-Based**: Learns transition and reward models from experience

#### RL Agent
- **Q-Learning**: Tabular Q-learning for smaller state spaces
- **Deep Q-Network (DQN)**: Neural network approximation for large state spaces
- **Experience Replay**: Efficient learning from past experiences
- **Epsilon-Greedy**: Balanced exploration-exploitation

## 📊 Outputs

The simulator generates several outputs:

### Console Output
- Real-time training progress
- Episode statistics
- Scheduling success rates
- Disruption handling metrics

### Visualizations (saved to `output/`)
- `mdp_training.png` - MDP agent training progress
- `rl_training.png` - RL agent training progress
- `comparison.png` - Agent performance comparison
- `schedule_heatmap.png` - Resource allocation heatmap
- `disruptions.png` - Disruption analysis
- `mdp_metrics.png` / `rl_metrics.png` - Detailed metrics

### Saved Models (saved to `models/`)
- `mdp_policy.pkl` - Trained MDP policy
- `rl_agent_model.pkl` - Trained RL model

## 🎯 Use Cases

1. **Educational Institutions**
   - School/university course scheduling
   - Handling teacher absences and room conflicts
   - Optimizing classroom utilization

2. **Healthcare**
   - Operating room scheduling
   - Staff assignment with unexpected absences
   - Equipment allocation

3. **Manufacturing**
   - Production line scheduling
   - Resource allocation with machine failures
   - Worker assignment optimization

4. **Transportation**
   - Fleet scheduling
   - Driver assignment with absences
   - Route optimization with disruptions

## 🔬 Customization

### Adding New Disruption Types

Edit `src/environment/disruption_generator.py`:

```python
def _generate_custom_disruption(self, severity):
    """Generate a custom disruption"""
    return {
        'custom_param': value,
        'duration': duration,
        'reason': reason
    }
```

### Custom Reward Functions

Modify `src/utils/reward_calculator.py`:

```python
def calculate_custom_reward(self, state, action, info):
    """Calculate custom reward"""
    reward = 0.0
    # Add your reward logic
    return reward
```

### Different State Representations

Extend `src/utils/state_representation.py` to customize state encoding.

## 📈 Performance Tips

1. **For Small State Spaces** (< 1000 states):
   - Use tabular Q-learning
   - Faster convergence
   - No neural network overhead

2. **For Large State Spaces** (> 1000 states):
   - Enable Deep Q-Network by installing TensorFlow
   - Adjust network architecture in `rl_agent.py`
   - Use experience replay buffer

3. **Hyperparameter Tuning**:
   - Increase `discount_factor` for long-term planning
   - Adjust `learning_rate` for stability vs. speed
   - Tune `exploration_rate` for exploration-exploitation balance

## 🐛 Troubleshooting

**Issue**: TensorFlow import errors
- **Solution**: DQN is optional. The simulator falls back to tabular Q-learning automatically.

**Issue**: Slow training
- **Solution**: Reduce `num_episodes` or `num_classes` in config, or enable parallel processing.

**Issue**: Poor performance
- **Solution**: Increase training episodes, adjust reward weights, or tune hyperparameters.

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*
- Bellman, R. (1957). *Dynamic Programming*

## 📝 License

This project is open source and available for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 🙏 Acknowledgments

Built with ❤️ for intelligent scheduling optimization using Reinforcement Learning and Markov Decision Processes.

---

**Happy Scheduling! 🎓📅**

## Project Structure

```
rl-scheduling-simulator
├── src
│   ├── main.py                  # Entry point for the simulation
│   ├── environment              # Contains the scheduling environment and disruption generator
│   ├── agents                   # Contains the MDP and RL agents
│   ├── models                   # Contains the Q-learning and policy network models
│   ├── utils                    # Utility functions for state representation and reward calculation
│   └── visualization             # Visualization tools for analyzing simulation results
├── tests                        # Unit tests for the environment, agents, and models
├── config                       # Configuration settings for the simulation
├── data                         # Sample scheduling data for testing
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd rl-scheduling-simulator
pip install -r requirements.txt
```

## Usage

To run the scheduling simulation, execute the following command:

```bash
python src/main.py
```

The simulation will initialize the environment, agents, and run the scheduling process based on the configurations defined in `config/simulation_config.yaml`.

## Features

- **Dynamic Scheduling**: Adapts to real-time disruptions in the scheduling process.
- **MDP and RL Algorithms**: Implements both MDP and RL approaches for decision-making.
- **Visualization**: Provides tools to visualize agent performance and scheduling efficiency.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.