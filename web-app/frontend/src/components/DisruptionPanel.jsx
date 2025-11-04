import React, { useState } from 'react';
import { AlertTriangle, UserX, DoorClosed, Users } from 'lucide-react';
import './DisruptionPanel.css';

function DisruptionPanel({ onInjectDisruption, disabled }) {
  const [selectedType, setSelectedType] = useState('teacher_absent');
  const [targetId, setTargetId] = useState('');

  const disruptionTypes = [
    {
      type: 'teacher_absent',
      label: 'Teacher Absence',
      icon: <UserX size={20} />,
      description: 'Mark a teacher as unavailable',
      color: '#EF4444'
    },
    {
      type: 'room_unavailable',
      label: 'Room Unavailable',
      icon: <DoorClosed size={20} />,
      description: 'Mark a room as temporarily closed',
      color: '#F59E0B'
    },
    {
      type: 'enrollment_change',
      label: 'Enrollment Change',
      icon: <Users size={20} />,
      description: 'Change student enrollment for a class',
      color: '#8B5CF6'
    }
  ];

  const handleInject = () => {
    if (!targetId || targetId === '') {
      alert('Please enter a target ID');
      return;
    }

    onInjectDisruption({
      type: selectedType,
      target_id: parseInt(targetId)
    });

    setTargetId('');
  };

  const selectedDisruption = disruptionTypes.find(d => d.type === selectedType);

  return (
    <div className="disruption-panel card">
      <h2>⚡ Inject Disruption</h2>
      <p className="panel-description">
        Manually trigger disruptions to test the agent's ability to adapt in real-time
      </p>

      <div className="disruption-types">
        {disruptionTypes.map(disruption => (
          <button
            key={disruption.type}
            className={`disruption-type-btn ${selectedType === disruption.type ? 'active' : ''}`}
            onClick={() => setSelectedType(disruption.type)}
            disabled={disabled}
            style={{
              borderColor: selectedType === disruption.type ? disruption.color : '#E5E7EB',
              background: selectedType === disruption.type ? `${disruption.color}15` : 'white'
            }}
          >
            <div className="disruption-icon" style={{ color: disruption.color }}>
              {disruption.icon}
            </div>
            <div className="disruption-info">
              <div className="disruption-label">{disruption.label}</div>
              <div className="disruption-description">{disruption.description}</div>
            </div>
          </button>
        ))}
      </div>

      {selectedDisruption && (
        <div className="inject-form">
          <div className="selected-disruption" style={{ borderLeftColor: selectedDisruption.color }}>
            <div className="disruption-header">
              <span style={{ color: selectedDisruption.color }}>{selectedDisruption.icon}</span>
              <span className="disruption-title">{selectedDisruption.label}</span>
            </div>
            
            <div className="form-group">
              <label>
                Target ID
                <input
                  type="number"
                  value={targetId}
                  onChange={(e) => setTargetId(e.target.value)}
                  placeholder={
                    selectedType === 'teacher_absent' ? 'Enter teacher ID' :
                    selectedType === 'room_unavailable' ? 'Enter room ID' :
                    'Enter class ID'
                  }
                  disabled={disabled}
                  min="0"
                />
              </label>
              <small>
                {selectedType === 'teacher_absent' && 'ID of the teacher who becomes unavailable'}
                {selectedType === 'room_unavailable' && 'ID of the room that becomes unavailable'}
                {selectedType === 'enrollment_change' && 'ID of the class with enrollment changes'}
              </small>
            </div>

            <button
              className="danger"
              onClick={handleInject}
              disabled={disabled || !targetId}
              style={{ width: '100%' }}
            >
              <AlertTriangle size={16} />
              Inject Disruption
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default DisruptionPanel;
