# RL Scheduling Simulator - Web Application

A real-time, interactive web-based simulation of a university scheduling system that uses Reinforcement Learning to dynamically adapt to disruptions.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Install Backend Dependencies**
```bash
cd web-app/backend
pip install -r requirements.txt
```

2. **Install Frontend Dependencies**
```bash
cd web-app/frontend
npm install
```

### Running the Application

**Option 1: Use the startup script (Recommended)**

Windows:
```bash
.\start-web-app.bat
```

Linux/Mac:
```bash
./start-web-app.sh
```

**Option 2: Manual startup**

Terminal 1 - Backend:
```bash
cd web-app/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 - Frontend:
```bash
cd web-app/frontend
npm run dev
```

Then open your browser to: http://localhost:3000

## 🎮 Features

- **Real-time Scheduling**: Watch the RL agent make decisions in real-time
- **Interactive Controls**: Start, pause, step through the simulation
- **Disruption Injection**: Manually inject teacher absences, room conflicts, and enrollment changes
- **Live Metrics**: View performance graphs and statistics as they update
- **Dynamic Timetable**: See the schedule adapt to constraints and disruptions
- **WebSocket Updates**: Real-time bidirectional communication between frontend and backend

## 📊 How to Use

1. **Configure Simulation**: Set the number of classes, teachers, rooms, and disruption probability
2. **Choose Agent Type**: Select between Q-Learning (RL) or Value Iteration (MDP)
3. **Adjust Parameters**: Tune learning rate, discount factor, and exploration rate
4. **Create Simulation**: Click the button to initialize the environment
5. **Control Playback**: Use auto-play for continuous execution or step manually
6. **Inject Disruptions**: Test the agent by adding unexpected events
7. **Monitor Performance**: Track rewards, conflicts, and scheduling success

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance async REST API
- **WebSockets**: Real-time bidirectional communication
- **NumPy/Pandas**: Data processing and state management
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: Modern UI framework
- **Vite**: Lightning-fast build tool
- **Recharts**: Interactive data visualization
- **Axios**: HTTP client for API requests
- **Lucide React**: Beautiful icon library

## 📁 Project Structure

```
web-app/
├── backend/
│   ├── main.py              # FastAPI server with REST API and WebSocket
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── src/
    │   ├── components/      # React components
    │   │   ├── ConfigPanel.jsx
    │   │   ├── TimetableGrid.jsx
    │   │   ├── DisruptionPanel.jsx
    │   │   ├── MetricsPanel.jsx
    │   │   └── ControlPanel.jsx
    │   ├── App.jsx          # Main application
    │   └── main.jsx         # Entry point
    ├── package.json         # Node.js dependencies
    └── vite.config.js       # Vite configuration
```

## 🔧 API Endpoints

- `POST /simulation/create` - Create a new simulation
- `GET /simulation/{sim_id}/state` - Get current state
- `POST /simulation/{sim_id}/step` - Execute one step
- `POST /simulation/{sim_id}/inject-disruption` - Inject a disruption
- `GET /simulation/{sim_id}/metrics` - Get performance metrics
- `DELETE /simulation/{sim_id}` - Delete a simulation
- `GET /simulations` - List all simulations
- `WS /ws/{sim_id}` - WebSocket for real-time updates

## 🐛 Troubleshooting

**Backend won't start:**
- Ensure Python 3.8+ is installed
- Check that port 8000 is not in use
- Verify all dependencies are installed

**Frontend won't start:**
- Ensure Node.js 16+ is installed
- Try deleting `node_modules` and running `npm install` again
- Check that port 3000 is not in use

**Can't connect to backend:**
- Verify the backend is running on port 8000
- Check CORS settings in `main.py`
- Ensure no firewall is blocking the connection

## 📝 License

MIT License - See LICENSE file for details
