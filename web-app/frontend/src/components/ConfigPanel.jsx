import React, { useState } from 'react';
import './ConfigPanel.css';

function ConfigPanel({ onCreateSimulation, loading, disabled }) {
  const [config, setConfig] = useState({
    num_classes: 10,
    num_teachers: 5,
    num_rooms: 3,
    disruption_probability: 0.1,  // Reduced from 0.2 to 0.1 (10%)
    agent_type: 'rl',
    learning_rate: 0.1,  // Increased from 0.01 to 0.1 for faster learning
    discount_factor: 0.95,  // Changed from 0.99 to 0.95
    epsilon: 0.5  // Increased from 0.3 to 0.5 for better exploration
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onCreateSimulation(config);
  };

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  return (
    <div className="config-panel card">
      <h2>⚙️ Simulation Configuration</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Environment Settings</h3>
          
          <div className="form-group">
            <label>
              Classes
              <input
                type="number"
                name="num_classes"
                value={config.num_classes}
                onChange={handleChange}
                min="1"
                max="50"
                disabled={disabled}
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Teachers
              <input
                type="number"
                name="num_teachers"
                value={config.num_teachers}
                onChange={handleChange}
                min="1"
                max="20"
                disabled={disabled}
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Rooms
              <input
                type="number"
                name="num_rooms"
                value={config.num_rooms}
                onChange={handleChange}
                min="1"
                max="10"
                disabled={disabled}
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Disruption Probability
              <input
                type="number"
                name="disruption_probability"
                value={config.disruption_probability}
                onChange={handleChange}
                min="0"
                max="1"
                step="0.1"
                disabled={disabled}
              />
              <small>{(config.disruption_probability * 100).toFixed(0)}% chance per step</small>
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3>Agent Settings</h3>
          
          <div className="form-group">
            <label>
              Agent Type
              <select
                name="agent_type"
                value={config.agent_type}
                onChange={handleChange}
                disabled={disabled}
              >
                <option value="rl">Q-Learning (RL)</option>
                <option value="mdp">Value Iteration (MDP)</option>
              </select>
            </label>
          </div>

          <div className="form-group">
            <label>
              Learning Rate
              <input
                type="number"
                name="learning_rate"
                value={config.learning_rate}
                onChange={handleChange}
                min="0.001"
                max="1"
                step="0.01"
                disabled={disabled}
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Discount Factor (γ)
              <input
                type="number"
                name="discount_factor"
                value={config.discount_factor}
                onChange={handleChange}
                min="0"
                max="1"
                step="0.01"
                disabled={disabled}
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Exploration Rate (ε)
              <input
                type="number"
                name="epsilon"
                value={config.epsilon}
                onChange={handleChange}
                min="0"
                max="1"
                step="0.1"
                disabled={disabled}
              />
            </label>
          </div>
        </div>

        <button
          type="submit"
          className="primary"
          disabled={disabled || loading}
          style={{ width: '100%', padding: '12px' }}
        >
          {loading ? '⏳ Creating...' : disabled ? '✅ Simulation Active' : '🚀 Create Simulation'}
        </button>
      </form>
    </div>
  );
}

export default ConfigPanel;
