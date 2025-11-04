import React from 'react';
import { Calendar } from 'lucide-react';
import './TimetableGrid.css';

function TimetableGrid({ state }) {
  // Time slot labels (8am to 4pm)
  const timeLabels = [
    '8:00-9:00',
    '9:00-10:00',
    '10:00-11:00',
    '11:00-12:00',
    '12:00-1:00',
    '1:00-2:00',
    '2:00-3:00',
    '3:00-4:00'
  ];

  if (!state || !state.schedule) {
    return (
      <div className="timetable-grid card">
        <div className="empty-state">
          <Calendar size={48} />
          <p>No schedule available yet</p>
          <small>Create a simulation to see the timetable</small>
        </div>
      </div>
    );
  }

  const { schedule, num_teachers, num_rooms, num_time_slots = 8 } = state;
  
  // Convert schedule dictionary to grid format
  const getScheduleGrid = () => {
    const grid = {};
    
    // Initialize empty grid
    for (let timeSlot = 0; timeSlot < num_time_slots; timeSlot++) {
      grid[timeSlot] = {};
      for (let room = 0; room < num_rooms; room++) {
        grid[timeSlot][room] = null;
      }
    }

    // Fill grid with scheduled classes from the schedule dictionary
    Object.keys(schedule).forEach(timeSlot => {
      const slotNum = parseInt(timeSlot);
      const classes = schedule[timeSlot];
      
      if (Array.isArray(classes)) {
        classes.forEach(classItem => {
          const roomId = classItem.room_id;
          if (roomId >= 0 && roomId < num_rooms) {
            grid[slotNum][roomId] = {
              classId: classItem.class_id,
              teacherId: classItem.teacher_id,
              roomId: roomId,
              status: classItem.status,
              students: classItem.students
            };
          }
        });
      }
    });

    return grid;
  };

  const scheduleGrid = getScheduleGrid();

  // Get cell color based on status
  const getCellClass = (cellData) => {
    if (!cellData || cellData.teacherId === -1 || cellData.roomId === -1) {
      return 'cell empty';
    }
    
    switch (cellData.status) {
      case 1: return 'cell scheduled';
      case -1: return 'cell disrupted';
      case 0: return 'cell unscheduled';
      default: return 'cell';
    }
  };

  // Format cell content
  const getCellContent = (cellData) => {
    if (!cellData || cellData.teacherId === -1 || cellData.roomId === -1) {
      return <span className="empty-text">Empty</span>;
    }

    return (
      <div className="cell-content">
        <div className="class-name">Class {cellData.classId}</div>
        <div className="teacher-name">Teacher {cellData.teacherId}</div>
        <div className="students-info">{cellData.students} students</div>
        <div className="status-badge">
          {cellData.status === 1 ? '✓ Scheduled' : 
           cellData.status === -1 ? '✗ Disrupted' : '○ Pending'}
        </div>
      </div>
    );
  };

  return (
    <div className="timetable-grid card">
      <h2>📅 Live Timetable</h2>
      
      <div className="grid-container">
        <div className="grid-wrapper">
          {/* Header row with room labels */}
          <div className="grid-header">
            <div className="header-cell time-header">Time Slot</div>
            {Array.from({ length: num_rooms }, (_, i) => (
              <div key={i} className="header-cell room-header">
                Room {i}
              </div>
            ))}
          </div>

          {/* Grid rows */}
          {Array.from({ length: num_time_slots }, (_, timeSlot) => (
            <div key={timeSlot} className="grid-row">
              <div className="time-cell">
                <div className="time-label">{timeLabels[timeSlot] || `Slot ${timeSlot}`}</div>
                <div className="slot-number">Slot {timeSlot}</div>
              </div>
              {Array.from({ length: num_rooms }, (_, room) => (
                <div 
                  key={room} 
                  className={getCellClass(scheduleGrid[timeSlot]?.[room])}
                >
                  {getCellContent(scheduleGrid[timeSlot]?.[room])}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="legend">
        <div className="legend-item">
          <span className="legend-color scheduled"></span>
          <span>Scheduled</span>
        </div>
        <div className="legend-item">
          <span className="legend-color disrupted"></span>
          <span>Disrupted</span>
        </div>
        <div className="legend-item">
          <span className="legend-color empty"></span>
          <span>Empty</span>
        </div>
      </div>

      <div className="timetable-summary">
        <div className="summary-item">
          <span className="summary-label">Teachers:</span>
          <span className="summary-value">{num_teachers}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Rooms:</span>
          <span className="summary-value">{num_rooms}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Time Slots:</span>
          <span className="summary-value">{num_time_slots}</span>
        </div>
      </div>
    </div>
  );
}

export default TimetableGrid;
