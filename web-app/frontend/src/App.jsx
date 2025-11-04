import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ConfigPanel from './components/ConfigPanel';
import TimetableGrid from './components/TimetableGrid';
import DisruptionPanel from './components/DisruptionPanel';
import MetricsPanel from './components/MetricsPanel';
import ControlPanel from './components/ControlPanel';
import ResultsSummary from './components/ResultsSummary';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [simId, setSimId] = useState(null);
  const [simulationState, setSimulationState] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [speed, setSpeed] = useState(2000); // 2 seconds default
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [maxSteps] = useState(200);
  
  const autoPlayInterval = useRef(null);
  const wsRef = useRef(null);

  // Create simulation
  const createSimulation = async (config) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE}/simulation/create`, config);
      setSimId(response.data.sim_id);
      await fetchSimulationState(response.data.sim_id);
      await fetchMetrics(response.data.sim_id);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch simulation state
  const fetchSimulationState = async (id) => {
    try {
      const response = await axios.get(`${API_BASE}/simulation/${id || simId}/state`);
      setSimulationState(response.data);
    } catch (err) {
      console.error('Error fetching state:', err);
    }
  };

  // Fetch metrics
  const fetchMetrics = async (id) => {
    try {
      const response = await axios.get(`${API_BASE}/simulation/${id || simId}/metrics`);
      setMetrics(response.data);
      return response.data;
    } catch (err) {
      console.error('Error fetching metrics:', err);
      return null;
    }
  };

  // Execute single step
  const executeStep = async () => {
    if (!simId) return;
    
    try {
      await axios.post(`${API_BASE}/simulation/${simId}/step`, { auto: true });
      await fetchSimulationState();
      const metricsData = await fetchMetrics();
      
      // Check if we've reached max steps
      if (metricsData && metricsData.length >= maxSteps) {
        setAutoPlay(false);
        setShowResults(true);
      }
    } catch (err) {
      console.error('Error executing step:', err);
    }
  };

  // Inject disruption
  const injectDisruption = async (disruption) => {
    if (!simId) return;
    
    try {
      await axios.post(`${API_BASE}/simulation/${simId}/inject-disruption`, disruption);
      await fetchSimulationState();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  // Auto-play functionality
  useEffect(() => {
    if (autoPlay && simId) {
      autoPlayInterval.current = setInterval(() => {
        executeStep();
      }, speed); // Use dynamic speed
    } else {
      if (autoPlayInterval.current) {
        clearInterval(autoPlayInterval.current);
      }
    }

    return () => {
      if (autoPlayInterval.current) {
        clearInterval(autoPlayInterval.current);
      }
    };
  }, [autoPlay, simId, speed]); // Add speed to dependencies

  // Reset simulation
  const resetSimulation = () => {
    setSimId(null);
    setSimulationState(null);
    setMetrics(null);
    setAutoPlay(false);
    setIsRunning(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🎓 RL-Based Dynamic Scheduling Simulator</h1>
        <p>Real-time University Timetable Optimization with Stochastic Disruptions</p>
      </header>

      {error && (
        <div className="error-banner">
          <strong>Error:</strong> {error}
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="main-layout">
        {/* Left Sidebar - Configuration */}
        <aside className="sidebar">
          <ConfigPanel
            onCreateSimulation={createSimulation}
            loading={loading}
            disabled={!!simId}
          />
          
          {simId && (
            <>
              <ControlPanel
                simId={simId}
                isPlaying={autoPlay}
                onTogglePlay={() => setAutoPlay(!autoPlay)}
                onStep={executeStep}
                onReset={resetSimulation}
                onDelete={resetSimulation}
                onSpeedChange={setSpeed}
                onShowResults={() => setShowResults(true)}
                speed={speed}
                disabled={loading}
                currentStep={state?.step || 0}
                currentEpisode={state?.episode || 0}
                maxEpisodes={200}
              />
              
              <DisruptionPanel
                onInjectDisruption={injectDisruption}
                disabled={loading}
              />
            </>
          )}
        </aside>

        {/* Main Content Area */}
        <main className="main-content">
          {!simId ? (
            <div className="welcome-screen card">
              <h2>Welcome to the RL Scheduling Simulator</h2>
              <p>Configure your simulation parameters in the left panel and click "Create Simulation" to begin.</p>
              <div className="feature-list">
                <div className="feature-item">
                  <span className="feature-icon">⚡</span>
                  <div>
                    <h3>Real-Time Optimization</h3>
                    <p>Watch the RL agent dynamically reassign resources</p>
                  </div>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🎲</span>
                  <div>
                    <h3>Stochastic Disruptions</h3>
                    <p>Inject random events and see how the system adapts</p>
                  </div>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">📊</span>
                  <div>
                    <h3>Live Metrics</h3>
                    <p>Track performance and scheduling efficiency</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              <TimetableGrid state={simulationState} />
              
              <MetricsPanel 
                metrics={metrics} 
                state={simulationState}
              />
            </>
          )}
        </main>
      </div>

      {/* Results Summary Modal */}
      {showResults && simId && (
        <ResultsSummary 
          simId={simId} 
          onClose={() => setShowResults(false)} 
        />
      )}
    </div>
  );
}

export default App;
