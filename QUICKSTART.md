# Quick Start Guide

Welcome to the RL Scheduling Simulator! This guide will help you get started quickly.

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Demo

**Windows:**
```bash
run.bat
```
Then select option 1 for Quick Demo.

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```
Then select option 1 for Quick Demo.

**Or directly:**
```bash
python demo.py
```

### Step 3: Run Full Simulation

**Windows:**
```bash
cd src
python main.py
```

**Linux/Mac:**
```bash
cd src
python3 main.py
```

Then select a simulation mode:
- **1**: MDP Only
- **2**: RL Only
- **3**: Hybrid (MDP + RL)
- **4**: Compare MDP vs RL

## 📊 What You'll See

### During Training
- Real-time progress updates
- Episode rewards
- Scheduling success rates
- Disruption counts

### After Training
- Performance plots (saved in `output/`)
- Schedule heatmaps
- Disruption analysis
- Trained models (saved in `models/`)

## ⚙️ Configuration

Edit `config/simulation_config.yaml` to customize:

```yaml
environment:
  num_classes: 10      # Change number of classes
  num_teachers: 5      # Change number of teachers
  num_rooms: 3         # Change number of rooms
  disruption_probability: 0.2  # Adjust disruption rate

simulation:
  num_episodes: 200    # Increase for more training
  max_steps_per_episode: 50
```

## 📈 Understanding the Output

### Rewards
- **Positive rewards**: Successful scheduling actions
- **Negative rewards**: Invalid actions or conflicts
- **Higher average reward** = Better performance

### Metrics
- **Scheduled Classes**: Number of successfully scheduled classes
- **Disruptions**: Number of disruptions encountered
- **Success Rate**: Percentage of successful actions

### Visualizations
- **Training Progress**: Shows learning curve
- **Schedule Heatmap**: Shows resource allocation
- **Disruption Analysis**: Shows disruption patterns
- **Comparison Plot**: Compares different agents

## 🎯 Try These Examples

### Example 1: Small Scale Test
```yaml
environment:
  num_classes: 5
  num_teachers: 3
  num_rooms: 2
  
simulation:
  num_episodes: 50
```

### Example 2: High Disruption Scenario
```yaml
environment:
  num_classes: 10
  disruption_probability: 0.5  # 50% chance of disruption!

simulation:
  num_episodes: 300
```

### Example 3: Large Scale
```yaml
environment:
  num_classes: 20
  num_teachers: 10
  num_rooms: 5
  
simulation:
  num_episodes: 500
```

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "TensorFlow not found" (optional)
The simulator works fine without TensorFlow. It automatically uses tabular Q-learning.

To enable Deep Q-Network (optional):
```bash
pip install tensorflow>=2.8.0
```

### Plots not showing
Make sure matplotlib backend is working:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Training too slow
- Reduce `num_episodes` in config
- Reduce `num_classes` in config
- Use MDP agent (faster than RL for small problems)

## 💡 Tips

1. **Start small**: Begin with 5 classes and 50 episodes
2. **Monitor progress**: Watch the console output
3. **Check visualizations**: Look at the plots in `output/`
4. **Experiment**: Try different configurations
5. **Compare agents**: Use mode 4 to see which works best

## 📚 Next Steps

After running the basic simulation:

1. **Customize disruptions** in `src/environment/disruption_generator.py`
2. **Adjust rewards** in `src/utils/reward_calculator.py`
3. **Add new features** to the state representation
4. **Experiment with hyperparameters** in the config file
5. **Try different algorithms** by modifying the agents

## 🎓 Learn More

- Read the full [README.md](README.md) for detailed documentation
- Explore the code in `src/` directory
- Check out the visualization tools in `src/visualization/`
- Review the configuration options in `config/`

## 🆘 Need Help?

- Check the console output for error messages
- Review the configuration file for typos
- Ensure all dependencies are installed
- Start with the demo to verify everything works

---

**Happy Scheduling! 🎉**
