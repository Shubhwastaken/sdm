# Troubleshooting Guide

This guide helps you resolve common issues when running the RL Scheduling Simulator.

## 🔴 Installation Issues

### Problem: "pip: command not found"
**Solution:**
- Make sure Python is installed and added to PATH
- Try `python -m pip install -r requirements.txt` instead

### Problem: "Permission denied" when installing packages
**Solution:**
- Windows: Run Command Prompt as Administrator
- Linux/Mac: Use `pip install --user -r requirements.txt`
- Or use a virtual environment:
  ```bash
  python -m venv venv
  # Windows:
  venv\Scripts\activate
  # Linux/Mac:
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### Problem: "Module 'numpy' not found" or similar
**Solution:**
```bash
pip install numpy pandas matplotlib seaborn pyyaml
```

## 🔴 Import Errors

### Problem: "Import 'environment.scheduling_env' could not be resolved"
**Solution:**
- These are just IDE warnings, the code will run fine
- The scripts add the correct paths automatically
- Or run from the correct directory:
  ```bash
  cd src
  python main.py
  ```

### Problem: "No module named 'tensorflow'"
**Solution:**
- TensorFlow is **optional** for Deep Q-Network
- The simulator automatically uses tabular Q-learning without it
- To enable DQN (optional):
  ```bash
  pip install tensorflow>=2.8.0
  ```

## 🔴 Runtime Errors

### Problem: "FileNotFoundError: config/simulation_config.yaml"
**Solution:**
- Make sure you're running from the project root directory:
  ```bash
  cd rl-scheduling-simulator
  python demo.py
  ```
- Or for main simulation:
  ```bash
  cd rl-scheduling-simulator/src
  python main.py
  ```

### Problem: "No valid actions available"
**Solution:**
- This can happen when all resources are occupied
- Increase the number of teachers or rooms in config:
  ```yaml
  environment:
    num_teachers: 5  # Increase this
    num_rooms: 3     # Or increase this
  ```

### Problem: Training is very slow
**Solutions:**
1. Reduce the number of episodes:
   ```yaml
   simulation:
     num_episodes: 50  # Start with fewer episodes
   ```

2. Reduce the problem size:
   ```yaml
   environment:
     num_classes: 5  # Fewer classes
   ```

3. Use MDP agent instead of RL (faster for small problems)

4. Disable visualizations temporarily by commenting out plot calls in main.py

## 🔴 Visualization Issues

### Problem: Plots don't show up
**Solutions:**
1. Check if matplotlib backend is working:
   ```python
   import matplotlib
   print(matplotlib.get_backend())
   ```

2. Try changing backend (add to beginning of script):
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'
   ```

3. Install additional dependencies:
   ```bash
   pip install python-tk  # Linux
   ```

### Problem: "RuntimeError: main thread is not in main loop"
**Solution:**
- This is a Matplotlib threading issue
- Close any open plot windows before running again
- Or use `matplotlib.use('Agg')` to save plots without displaying

## 🔴 Performance Issues

### Problem: Very low or negative rewards throughout training
**Solutions:**
1. Increase training duration:
   ```yaml
   num_episodes: 500  # More training
   ```

2. Adjust learning rate:
   ```yaml
   agents:
     rl:
       learning_rate: 0.01  # Try different values: 0.001, 0.01, 0.1
   ```

3. Adjust exploration:
   ```yaml
   agents:
     rl:
       epsilon: 1.0  # Start with full exploration
   ```

4. Reduce disruption rate:
   ```yaml
   environment:
     disruption_probability: 0.1  # Lower disruptions
   ```

### Problem: Agent not improving
**Solutions:**
1. Check reward function weights in `reward_calculator.py`
2. Ensure enough exploration (epsilon not too low)
3. Verify environment constraints aren't too strict
4. Try the MDP agent instead for comparison

## 🔴 Configuration Issues

### Problem: "yaml.scanner.ScannerError"
**Solution:**
- Check YAML syntax in `config/simulation_config.yaml`
- Ensure proper indentation (use spaces, not tabs)
- Verify no special characters in values
- Example of correct format:
  ```yaml
  environment:
    num_classes: 10
    num_teachers: 5
  ```

### Problem: "KeyError" when accessing config
**Solution:**
- Ensure all required fields are in config file
- Compare with the default `simulation_config.yaml`
- Missing fields should have default values in code

## 🔴 Memory Issues

### Problem: "MemoryError" or system freezes
**Solutions:**
1. Reduce state space size:
   ```yaml
   environment:
     num_classes: 5
     num_teachers: 3
   ```

2. Limit experience replay buffer (in `rl_agent.py`):
   ```python
   self.memory = deque(maxlen=1000)  # Reduce from 10000
   ```

3. Use tabular Q-learning instead of DQN
4. Close other applications

## 🔴 Output Issues

### Problem: No output files generated
**Solutions:**
1. Check if `output/` directory exists
2. Verify write permissions
3. Look for error messages in console
4. Try running with administrator/sudo privileges

### Problem: Can't save models
**Solutions:**
1. Create `models/` directory manually:
   ```bash
   mkdir models
   ```
2. Check disk space
3. Verify write permissions

## 🔴 Simulation Issues

### Problem: Episode ends too quickly
**Solutions:**
1. Increase max steps:
   ```yaml
   simulation:
     max_steps_per_episode: 100  # Increase from 50
   ```

2. Check if all classes are being scheduled prematurely
3. Review termination conditions in `scheduling_env.py`

### Problem: Too many disruptions
**Solution:**
- Reduce disruption probability:
  ```yaml
  environment:
    disruption_probability: 0.1  # Reduce from 0.2
  ```

### Problem: Not enough valid actions
**Solutions:**
1. Increase resources:
   ```yaml
   environment:
     num_teachers: 7  # More teachers
     num_rooms: 4     # More rooms
   ```

2. Increase time slots (modify in code):
   ```python
   self.num_time_slots = 16  # Increase from 8
   ```

## 🔴 Code Modification Issues

### Problem: Changes not taking effect
**Solutions:**
1. Restart Python interpreter
2. Clear `__pycache__` directories:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +  # Linux/Mac
   # Or manually delete __pycache__ folders
   ```
3. Ensure you're editing the correct file
4. Check for syntax errors

### Problem: Custom modifications cause errors
**Solutions:**
1. Check Python syntax
2. Verify indentation (use 4 spaces)
3. Import necessary modules
4. Revert to original code and make changes incrementally
5. Use print statements to debug

## 🟢 Verification Steps

To verify everything is working:

1. **Test imports:**
   ```bash
   python -c "import numpy, pandas, matplotlib, seaborn, yaml"
   ```

2. **Run demo:**
   ```bash
   python demo.py
   ```

3. **Check output:**
   - Console shows progress
   - No error messages
   - Plots appear
   - Files created in `demo_output/`

## 📞 Getting Help

If you're still stuck:

1. **Check console output** - Error messages are helpful
2. **Review configuration** - Most issues are config-related
3. **Start simple** - Use small configurations first
4. **Check dependencies** - Ensure all packages installed
5. **Read error messages** - They usually point to the problem

## 🛠️ Debug Mode

Add this to any script for detailed debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add throughout code:
print(f"Debug: variable_name = {variable_name}")
```

## 🔄 Reset Everything

If things are really broken:

```bash
# Remove all generated files
rm -rf output/ models/ demo_output/ example_output/
rm -rf __pycache__/ src/__pycache__/ src/*/__pycache__/

# Reinstall dependencies
pip uninstall -y numpy pandas matplotlib seaborn pyyaml
pip install -r requirements.txt

# Run fresh demo
python demo.py
```

## ✅ Common Solutions Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Running from correct directory
- [ ] Config file has correct syntax
- [ ] No missing required fields in config
- [ ] Sufficient system resources (RAM, disk space)
- [ ] No conflicting Python versions
- [ ] Virtual environment activated (if using one)

---

**Still having issues?** Check the error message carefully - it usually tells you exactly what's wrong! 🔍
