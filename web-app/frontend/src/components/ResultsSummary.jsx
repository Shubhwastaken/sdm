import React, { useEffect, useState } from 'react';
import { Trophy, TrendingUp, Target, AlertCircle, CheckCircle, Award } from 'lucide-react';
import './ResultsSummary.css';

function ResultsSummary({ simId, onClose }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch(`http://localhost:8000/simulation/${simId}/summary`);
        const data = await response.json();
        setSummary(data);
      } catch (error) {
        console.error('Error fetching summary:', error);
      } finally {
        setLoading(false);
      }
    };

    if (simId) {
      fetchSummary();
    }
  }, [simId]);

  if (loading) {
    return (
      <div className="results-overlay">
        <div className="results-modal">
          <div className="loading-spinner">Loading results...</div>
        </div>
      </div>
    );
  }

  if (!summary) {
    return null;
  }

  const { statistics, performance, disruptions, resource_utilization, configuration } = summary;

  const getGrade = () => {
    if (performance.success_rate === 100 && performance.average_conflicts_per_step < 1) return 'A+';
    if (performance.success_rate >= 90) return 'A';
    if (performance.success_rate >= 80) return 'B';
    if (performance.success_rate >= 70) return 'C';
    return 'D';
  };

  const grade = getGrade();

  return (
    <div className="results-overlay">
      <div className="results-modal">
        {/* Header */}
        <div className="results-header">
          <Trophy size={48} className="trophy-icon" />
          <h1>🎓 Simulation Complete!</h1>
          <p>Training finished after {statistics.total_episodes} episodes</p>
        </div>

        {/* Grade Badge */}
        <div className={`grade-badge grade-${grade.replace('+', 'plus')}`}>
          <div className="grade-label">Overall Grade</div>
          <div className="grade-value">{grade}</div>
          <div className="grade-subtitle">
            {grade === 'A+' ? 'Excellent Performance!' :
             grade === 'A' ? 'Great Work!' :
             grade === 'B' ? 'Good Job!' :
             grade === 'C' ? 'Satisfactory' : 'Needs Improvement'}
          </div>
        </div>

        {/* Best Reward Section */}
        {summary.best_reward && (
          <div className="best-reward-section">
            <div className="best-reward-header">
              <Award size={36} className="best-reward-icon" />
              <div>
                <h2>🏆 Best Reward Achieved</h2>
                <p>Peak performance reached at step {summary.best_reward.step}</p>
              </div>
            </div>
            <div className="best-reward-content">
              <div className="best-reward-main">
                <div className="best-reward-value">{summary.best_reward.reward.toFixed(2)}</div>
                <div className="best-reward-label">Maximum Reward</div>
              </div>
              <div className="best-reward-characteristics">
                <h3>Characteristics of Best Performance:</h3>
                <div className="characteristics-grid">
                  <div className="characteristic-item">
                    <CheckCircle size={20} className="char-icon" />
                    <div>
                      <div className="char-value">{summary.best_reward.characteristics.scheduled_classes}</div>
                      <div className="char-label">Classes Scheduled</div>
                    </div>
                  </div>
                  <div className="characteristic-item">
                    <Target size={20} className="char-icon" />
                    <div>
                      <div className="char-value">{summary.best_reward.characteristics.success_rate.toFixed(1)}%</div>
                      <div className="char-label">Success Rate</div>
                    </div>
                  </div>
                  <div className="characteristic-item">
                    <AlertCircle size={20} className="char-icon" />
                    <div>
                      <div className="char-value">{summary.best_reward.characteristics.conflicts}</div>
                      <div className="char-label">Conflicts</div>
                    </div>
                  </div>
                  <div className="characteristic-item">
                    <TrendingUp size={20} className="char-icon" />
                    <div>
                      <div className="char-value">{summary.best_reward.characteristics.teacher_utilization.toFixed(0)}%</div>
                      <div className="char-label">Teacher Utilization</div>
                    </div>
                  </div>
                  <div className="characteristic-item">
                    <TrendingUp size={20} className="char-icon" />
                    <div>
                      <div className="char-value">{summary.best_reward.characteristics.room_utilization.toFixed(0)}%</div>
                      <div className="char-label">Room Utilization</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Main Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card primary">
            <div className="stat-icon">
              <TrendingUp size={32} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{statistics.average_reward.toFixed(2)}</div>
              <div className="stat-label">Average Reward</div>
              <div className="stat-sublabel">
                {statistics.improvement_percentage > 0 ? '+' : ''}
                {statistics.improvement_percentage.toFixed(1)}% improvement
              </div>
            </div>
          </div>

          <div className="stat-card success">
            <div className="stat-icon">
              <CheckCircle size={32} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{performance.success_rate.toFixed(0)}%</div>
              <div className="stat-label">Success Rate</div>
              <div className="stat-sublabel">
                {performance.classes_scheduled}/{performance.total_classes} classes scheduled
              </div>
            </div>
          </div>

          <div className="stat-card warning">
            <div className="stat-icon">
              <AlertCircle size={32} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{performance.total_conflicts}</div>
              <div className="stat-label">Total Conflicts</div>
              <div className="stat-sublabel">
                {performance.conflict_reduction.toFixed(0)}% reduction
              </div>
            </div>
          </div>

          <div className="stat-card info">
            <div className="stat-icon">
              <Target size={32} />
            </div>
            <div className="stat-content">
              <div className="stat-value">{disruptions.total_disruptions}</div>
              <div className="stat-label">Disruptions Handled</div>
              <div className="stat-sublabel">
                {disruptions.adaptation_rate.toFixed(0)}% adaptation rate
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Sections */}
        <div className="details-sections">
          {/* Training Progress */}
          <div className="detail-section">
            <h3>📊 Training Progress</h3>
            <div className="detail-rows">
              <div className="detail-row">
                <span>Total Episodes:</span>
                <strong>{statistics.total_episodes}</strong>
              </div>
              <div className="detail-row">
                <span>Total Steps:</span>
                <strong>{statistics.total_steps}</strong>
              </div>
              <div className="detail-row">
                <span>Best Episode Reward:</span>
                <strong>{statistics.max_reward.toFixed(2)}</strong>
              </div>
              <div className="detail-row">
                <span>Recent Avg (Last 50):</span>
                <strong>{statistics.recent_average_reward.toFixed(2)}</strong>
              </div>
              <div className="detail-row">
                <span>Total Cumulative Reward:</span>
                <strong>{statistics.total_reward.toFixed(2)}</strong>
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="detail-section">
            <h3>🎯 Performance Metrics</h3>
            <div className="detail-rows">
              <div className="detail-row">
                <span>Classes Scheduled:</span>
                <strong className="success-text">
                  {performance.classes_scheduled}/{performance.total_classes}
                </strong>
              </div>
              <div className="detail-row">
                <span>Scheduling Success:</span>
                <strong className="success-text">{performance.success_rate.toFixed(1)}%</strong>
              </div>
              <div className="detail-row">
                <span>Average Conflicts:</span>
                <strong>{performance.average_conflicts_per_step.toFixed(2)}</strong>
              </div>
              <div className="detail-row">
                <span>Conflict Reduction:</span>
                <strong className="success-text">{performance.conflict_reduction.toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          {/* Resource Utilization */}
          <div className="detail-section">
            <h3>💎 Resource Utilization</h3>
            <div className="detail-rows">
              <div className="detail-row">
                <span>Teacher Utilization:</span>
                <div className="progress-bar">
                  <div 
                    className="progress-fill teacher"
                    style={{ width: `${resource_utilization.teacher_utilization}%` }}
                  >
                    {resource_utilization.teacher_utilization.toFixed(0)}%
                  </div>
                </div>
              </div>
              <div className="detail-row">
                <span>Room Utilization:</span>
                <div className="progress-bar">
                  <div 
                    className="progress-fill room"
                    style={{ width: `${resource_utilization.room_utilization}%` }}
                  >
                    {resource_utilization.room_utilization.toFixed(0)}%
                  </div>
                </div>
              </div>
              <div className="detail-row">
                <span>Teachers Available:</span>
                <strong>{resource_utilization.teachers}</strong>
              </div>
              <div className="detail-row">
                <span>Rooms Available:</span>
                <strong>{resource_utilization.rooms}</strong>
              </div>
            </div>
          </div>

          {/* Configuration */}
          <div className="detail-section">
            <h3>⚙️ Configuration</h3>
            <div className="detail-rows">
              <div className="detail-row">
                <span>Agent Type:</span>
                <strong className="uppercase">{configuration.agent_type}</strong>
              </div>
              <div className="detail-row">
                <span>Classes:</span>
                <strong>{configuration.num_classes}</strong>
              </div>
              <div className="detail-row">
                <span>Teachers:</span>
                <strong>{configuration.num_teachers}</strong>
              </div>
              <div className="detail-row">
                <span>Rooms:</span>
                <strong>{configuration.num_rooms}</strong>
              </div>
              <div className="detail-row">
                <span>Time Slots:</span>
                <strong>{configuration.num_time_slots}</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Key Achievements */}
        <div className="achievements">
          <h3><Award size={20} /> Key Achievements</h3>
          <div className="achievement-list">
            {performance.success_rate === 100 && (
              <div className="achievement">
                <span className="achievement-icon">✅</span>
                <span>Perfect Scheduling - All classes scheduled!</span>
              </div>
            )}
            {statistics.improvement_percentage > 100 && (
              <div className="achievement">
                <span className="achievement-icon">📈</span>
                <span>Outstanding Learning - Over 100% improvement!</span>
              </div>
            )}
            {performance.conflict_reduction > 80 && (
              <div className="achievement">
                <span className="achievement-icon">🎯</span>
                <span>Conflict Master - 80%+ conflict reduction!</span>
              </div>
            )}
            {disruptions.total_disruptions > 0 && (
              <div className="achievement">
                <span className="achievement-icon">🛡️</span>
                <span>Disruption Handler - {disruptions.total_disruptions} disruptions handled!</span>
              </div>
            )}
            {resource_utilization.teacher_utilization > 80 && (
              <div className="achievement">
                <span className="achievement-icon">👨‍🏫</span>
                <span>Efficient Resource Use - High teacher utilization!</span>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="results-actions">
          <button className="btn-secondary" onClick={onClose}>
            Close
          </button>
          <button className="btn-primary" onClick={() => window.print()}>
            📄 Export Results
          </button>
        </div>
      </div>
    </div>
  );
}

export default ResultsSummary;
