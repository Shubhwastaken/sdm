import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, Target, Award, AlertCircle } from 'lucide-react';
import './MetricsPanel.css';

function MetricsPanel({ metrics, state }) {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="metrics-panel card">
        <h2>📊 Performance Metrics</h2>
        <div className="empty-state">
          <TrendingUp size={48} />
          <p>No metrics available yet</p>
          <small>Execute simulation steps to see performance data</small>
        </div>
      </div>
    );
  }

  // Get latest metrics
  const latest = metrics[metrics.length - 1];
  
  // Calculate statistics
  const avgReward = metrics.length > 0 
    ? metrics.reduce((sum, m) => sum + m.reward, 0) / metrics.length 
    : 0;
  
  const maxReward = metrics.length > 0 
    ? Math.max(...metrics.map(m => m.reward)) 
    : 0;
  
  const minReward = metrics.length > 0 
    ? Math.min(...metrics.map(m => m.reward)) 
    : 0;

  // Prepare chart data
  const chartData = metrics.map((m, idx) => ({
    step: idx + 1,
    reward: parseFloat(m.reward.toFixed(2)),
    conflicts: m.conflicts || 0,
    unscheduled: m.unscheduled_classes || 0
  }));

  // Stats cards data
  const statsCards = [
    {
      icon: <Award size={24} />,
      label: 'Current Reward',
      value: latest.reward.toFixed(2),
      color: '#10B981',
      trend: metrics.length > 1 ? (latest.reward - metrics[metrics.length - 2].reward).toFixed(2) : null
    },
    {
      icon: <TrendingUp size={24} />,
      label: 'Average Reward',
      value: avgReward.toFixed(2),
      color: '#3B82F6',
      subtext: `Range: ${minReward.toFixed(1)} - ${maxReward.toFixed(1)}`
    },
    {
      icon: <AlertCircle size={24} />,
      label: 'Conflicts',
      value: latest.conflicts || 0,
      color: latest.conflicts > 0 ? '#EF4444' : '#10B981',
      subtext: 'Current step'
    },
    {
      icon: <Target size={24} />,
      label: 'Unscheduled Classes',
      value: latest.unscheduled_classes || 0,
      color: latest.unscheduled_classes > 0 ? '#F59E0B' : '#10B981',
      subtext: 'Needs assignment'
    }
  ];

  return (
    <div className="metrics-panel card">
      <h2>📊 Performance Metrics</h2>
      
      <div className="stats-grid">
        {statsCards.map((stat, idx) => (
          <div key={idx} className="stat-card" style={{ borderLeftColor: stat.color }}>
            <div className="stat-icon" style={{ color: stat.color }}>
              {stat.icon}
            </div>
            <div className="stat-content">
              <div className="stat-label">{stat.label}</div>
              <div className="stat-value" style={{ color: stat.color }}>
                {stat.value}
                {stat.trend !== null && stat.trend !== undefined && (
                  <span className={`stat-trend ${parseFloat(stat.trend) >= 0 ? 'positive' : 'negative'}`}>
                    {parseFloat(stat.trend) >= 0 ? '↑' : '↓'} {Math.abs(stat.trend)}
                  </span>
                )}
              </div>
              {stat.subtext && <div className="stat-subtext">{stat.subtext}</div>}
            </div>
          </div>
        ))}
      </div>

      <div className="charts-container">
        <div className="chart-section">
          <h3>Reward Over Time</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis 
                dataKey="step" 
                stroke="#6B7280"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="#6B7280"
                style={{ fontSize: '12px' }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'white', 
                  border: '1px solid #E5E7EB',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
              <Legend 
                wrapperStyle={{ fontSize: '14px' }}
              />
              <Line 
                type="monotone" 
                dataKey="reward" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={{ fill: '#3B82F6', r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-section">
          <h3>Issues Detected</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis 
                dataKey="step" 
                stroke="#6B7280"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="#6B7280"
                style={{ fontSize: '12px' }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'white', 
                  border: '1px solid #E5E7EB',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
              <Legend 
                wrapperStyle={{ fontSize: '14px' }}
              />
              <Bar dataKey="conflicts" fill="#EF4444" name="Conflicts" />
              <Bar dataKey="unscheduled" fill="#F59E0B" name="Unscheduled" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="metrics-summary">
        <div className="summary-row">
          <span className="summary-label">Total Steps:</span>
          <span className="summary-value">{metrics.length}</span>
        </div>
        <div className="summary-row">
          <span className="summary-label">Best Reward:</span>
          <span className="summary-value highlight">{maxReward.toFixed(2)}</span>
        </div>
        <div className="summary-row">
          <span className="summary-label">Success Rate:</span>
          <span className="summary-value">
            {((metrics.filter(m => (m.conflicts || 0) === 0 && (m.unscheduled_classes || 0) === 0).length / metrics.length) * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

export default MetricsPanel;
