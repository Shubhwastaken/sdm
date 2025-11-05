import React, { useState, useEffect } from 'react';
import { Play, Pause, SkipForward, RotateCcw, Trash2, Gauge, Trophy } from 'lucide-react';
import './ControlPanel.css';

function ControlPanel({ 
  simId, 
  isPlaying, 
  onTogglePlay, 
  onStep, 
  onReset,
  onDelete,
  onSpeedChange,
  onShowResults,
  speed = 2000,
  disabled,
  currentStep = 0,
  currentEpisode = 0,
  maxEpisodes = 100
}) {
  // Calculate progress
  const episodeProgress = Math.min((currentEpisode / maxEpisodes) * 100, 100);
  const isComplete = currentEpisode >= maxEpisodes;

  // Auto-stop when max episodes reached
  useEffect(() => {
    if (isComplete && isPlaying) {
      console.log(`🎉 Training Complete! ${maxEpisodes} episodes reached.`);
      onTogglePlay(); // Stop auto-play
      
      // Show results automatically after 1 second
      setTimeout(() => {
        if (onShowResults) {
          onShowResults();
        }
      }, 1000);
    }
  }, [isComplete, isPlaying, maxEpisodes, onShowResults, onTogglePlay]);

  if (!simId) {
    return (
      <div className="control-panel card">
        <div className="empty-state">
          <p>No simulation running</p>
          <small>Create a simulation to start</small>
        </div>
      </div>
    );
  }

  return (
    <div className="control-panel card">
      <h2>🎮 Simulation Controls</h2>
      
      {/* Episode Progress Bar */}
      <div className="episode-progress">
        <div className="progress-header">
          <span className="progress-label">Episode Progress</span>
          <span className={`progress-count ${isComplete ? 'complete' : ''}`}>
            {currentEpisode} / {maxEpisodes}
            {isComplete && ' ✅'}
          </span>
        </div>
        <div className="progress-bar">
          <div 
            className={`progress-fill ${isComplete ? 'complete' : ''}`}
            style={{ width: `${episodeProgress}%` }}
          />
        </div>
        <div className="progress-percentage">
          {episodeProgress.toFixed(1)}% Complete
        </div>
      </div>

      {/* Status Display */}
      <div className="status-display">
        <div className="status-item">
          <span className="status-label">Current Episode:</span>
          <span className="status-value">{currentEpisode}</span>
        </div>
        <div className="status-item">
          <span className="status-label">Total Steps:</span>
          <span className="status-value">{currentStep}</span>
        </div>
        <div className="status-item">
          <span className="status-label">Status:</span>
          <span className={`status-badge ${isComplete ? 'complete' : isPlaying ? 'running' : 'paused'}`}>
            {isComplete ? '✅ Complete' : isPlaying ? '▶️ Running' : '⏸️ Paused'}
          </span>
        </div>
      </div>

      <div className="control-buttons">
        <button
          className={isPlaying ? 'warning' : 'primary'}
          onClick={onTogglePlay}
          disabled={disabled || isComplete}
        >
          {isPlaying ? (
            <>
              <Pause size={20} />
              Pause Auto-Play
            </>
          ) : isComplete ? (
            <>
              ✅ Training Complete
            </>
          ) : (
            <>
              <Play size={20} />
              Start Auto-Play
            </>
          )}
        </button>

        <button
          className="secondary"
          onClick={onStep}
          disabled={disabled || isPlaying || isComplete}
        >
          <SkipForward size={20} />
          Single Step
        </button>
      </div>

      <div className="divider"></div>

      {/* Completion Message */}
      {isComplete && (
        <div className="completion-message">
          <h3>🎉 Training Complete!</h3>
          <p>The agent has completed {maxEpisodes} episodes of training.</p>
          <p>Click "View Results Summary" to see the final performance metrics.</p>
        </div>
      )}

      <div className="speed-control">
        <label>
          <Gauge size={16} />
          <span>Simulation Speed</span>
        </label>
        <input
          type="range"
          min="500"
          max="5000"
          step="500"
          value={speed}
          onChange={(e) => onSpeedChange && onSpeedChange(parseInt(e.target.value))}
          disabled={disabled}
          className="speed-slider"
        />
        <div className="speed-label">
          {speed < 1000 ? 'Very Fast' :
           speed < 2000 ? 'Fast' :
           speed < 3000 ? 'Normal' :
           speed < 4000 ? 'Slow' : 'Very Slow'}
          <span className="speed-value">({speed}ms)</span>
        </div>
      </div>

      <div className="divider"></div>

      <div className="action-buttons">
        <button
          className="secondary"
          onClick={onShowResults}
          disabled={disabled}
        >
          <Trophy size={18} />
          View Results Summary
        </button>

        <button
          className="secondary"
          onClick={onReset}
          disabled={disabled}
        >
          <RotateCcw size={18} />
          Reset Simulation
        </button>

        <button
          className="danger"
          onClick={onDelete}
          disabled={disabled}
        >
          <Trash2 size={18} />
          Delete Simulation
        </button>
      </div>

      <div className="status-info">
        <div className="status-item">
          <span className="status-label">Simulation ID:</span>
          <span className="status-value">{simId.substring(0, 8)}...</span>
        </div>
      </div>

      <div className="info-box">
        <strong>💡 Tips:</strong>
        <ul>
          <li><strong>Auto-Play:</strong> Continuously executes steps at selected speed</li>
          <li><strong>Single Step:</strong> Execute one decision-making step manually</li>
          <li><strong>Episodes:</strong> Agent learns over {maxEpisodes} complete scheduling cycles</li>
          <li><strong>Progress:</strong> Watch the episode counter and success rate improve!</li>
        </ul>
      </div>
    </div>
  );
}

export default ControlPanel;
