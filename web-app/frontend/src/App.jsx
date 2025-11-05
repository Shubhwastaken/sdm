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

// Configure axios defaults
axios.defaults.timeout = 15000; // 15 second default timeout
axios.defaults.headers.post['Content-Type'] = 'application/json';

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
  
  const autoPlayInterval = useRef(null);
  const wsRef = useRef(null);

  // Create simulation
  const createSimulation = async (config) => {
    setLoading(true);
    setError(null);
    try {
      console.log('Creating simulation with config:', config);
      const response = await axios.post(`${API_BASE}/simulation/create`, config);
      console.log('Simulation created:', response.data);
      setSimId(response.data.sim_id);
      await fetchSimulationState(response.data.sim_id);
      await fetchMetrics(response.data.sim_id);
      console.log('Simulation initialized successfully');
    } catch (err) {
      console.error('Error creating simulation:', err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch simulation state
  const fetchSimulationState = async (id) => {
    try {
      const simIdToUse = id || simId;
      console.log('Fetching state for simulation:', simIdToUse);
      const response = await axios.get(`${API_BASE}/simulation/${simIdToUse}/state`, {
        timeout: 5000 // Shorter timeout for state fetches
      });
      console.log('State received:', response.data);
      setSimulationState(response.data);
    } catch (err) {
      console.error('Error fetching state:', err.message);
      // Don't throw, just log - we don't want state fetch failures to stop the simulation
    }
  };

  // Fetch metrics
  const fetchMetrics = async (id) => {
    try {
      const simIdToUse = id || simId;
      console.log('Fetching metrics for simulation:', simIdToUse);
      const response = await axios.get(`${API_BASE}/simulation/${simIdToUse}/metrics`, {
        timeout: 5000 // Shorter timeout for metrics fetches
      });
      console.log('Metrics received:', response.data.length, 'data points');
      setMetrics(response.data);
      return response.data;
    } catch (err) {
      console.error('Error fetching metrics:', err.message);
      // Don't throw, just log
      return null;
    }
  };

  // Execute single step
  const executeStep = async () => {
    if (!simId) return;
    
    try {
      console.log('Executing step for simulation:', simId);
      const response = await axios.post(`${API_BASE}/simulation/${simId}/step`, { auto: true });
      console.log('Step response:', response.data);
      await fetchSimulationState();
      await fetchMetrics();
    } catch (err) {
      console.error('Error executing step:', err);
      setError(err.response?.data?.detail || err.message);
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
      console.log('Starting auto-play with speed:', speed);
      
      let isRunning = true;
      
      const runStep = async () => {
        if (!isRunning) return;
        
        try {
          console.log('Auto-play executing step...');
          const stepResponse = await axios.post(
            `${API_BASE}/simulation/${simId}/step`, 
            { auto: true },
            { timeout: 10000 } // 10 second timeout
          );
          console.log('Step completed:', stepResponse.data);
          
          if (isRunning) {
            await fetchSimulationState(simId);
            await fetchMetrics(simId);
          }
        } catch (err) {
          console.error('Error in auto-play:', err);
          
          // Check if it's a network timeout or connection error
          if (err.code === 'ECONNABORTED' || err.code === 'ERR_NETWORK') {
            console.warn('Network timeout, continuing...');
            // Don't stop auto-play on timeout, just log and continue
            setError(null); // Clear any previous errors
          } else {
            console.error('Error details:', err.response?.data);
            setError(`Step error: ${err.response?.data?.detail || err.message}`);
            setAutoPlay(false); // Stop on real error
          }
        }
      };
      
      // Execute first step immediately
      runStep();
      
      // Then set up interval for subsequent steps
      autoPlayInterval.current = setInterval(runStep, speed);
      
      return () => {
        isRunning = false;
        if (autoPlayInterval.current) {
          clearInterval(autoPlayInterval.current);
          autoPlayInterval.current = null;
        }
      };
    } else {
      if (autoPlayInterval.current) {
        console.log('Stopping auto-play');
        clearInterval(autoPlayInterval.current);
        autoPlayInterval.current = null;
      }
    }
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
                currentStep={simulationState?.step || 0}
                currentEpisode={simulationState?.episode || 0}
                maxEpisodes={100}
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
