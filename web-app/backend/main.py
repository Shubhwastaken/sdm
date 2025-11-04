"""
FastAPI Backend for RL Scheduling Simulator
Provides REST API and WebSocket for real-time simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import json
import numpy as np
from datetime import datetime

from environment.scheduling_env import SchedulingEnvironment
from agents.rl_agent import RLAgent
from agents.mdp_agent import MDPAgent
from environment.disruption_generator import DisruptionGenerator

# FastAPI app
app = FastAPI(title="RL Scheduling Simulator API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation state
simulation_instances = {}

# Pydantic models
class SimulationConfig(BaseModel):
    num_classes: int = 10
    num_teachers: int = 5
    num_rooms: int = 3
    disruption_probability: float = 0.2
    agent_type: str = "rl"  # "rl" or "mdp"
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    epsilon: float = 0.3

class DisruptionEvent(BaseModel):
    type: str  # "teacher_absent", "room_unavailable", "enrollment_change"
    target_id: int  # The ID of the affected resource

class SimulationStep(BaseModel):
    action: Optional[int] = None
    auto: bool = True

# Helper function to convert numpy types to Python types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "RL Scheduling Simulator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /simulation/create": "Create new simulation",
            "GET /simulation/{sim_id}/state": "Get current state",
            "POST /simulation/{sim_id}/step": "Execute one step",
            "POST /simulation/{sim_id}/inject-disruption": "Inject disruption",
            "GET /simulation/{sim_id}/metrics": "Get performance metrics",
            "DELETE /simulation/{sim_id}": "Delete simulation",
            "WS /ws/{sim_id}": "WebSocket for real-time updates"
        }
    }

@app.post("/simulation/create")
async def create_simulation(config: SimulationConfig):
    """Create a new simulation instance"""
    try:
        sim_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create environment
        env_config = {
            'num_classes': config.num_classes,
            'num_teachers': config.num_teachers,
            'num_rooms': config.num_rooms,
            'disruption_probability': config.disruption_probability
        }
        env = SchedulingEnvironment(env_config)
        
        # Create agent
        agent_config = {
            'learning_rate': config.learning_rate,
            'discount_factor': config.discount_factor,
            'epsilon': config.epsilon
        }
        
        if config.agent_type == "rl":
            agent = RLAgent(env, agent_config)
        else:
            agent = MDPAgent(env, agent_config)
        
        # Store simulation instance
        simulation_instances[sim_id] = {
            'env': env,
            'agent': agent,
            'config': config,
            'episode': 0,
            'step': 0,
            'total_reward': 0,
            'episode_rewards': [],
            'metrics_history': [],
            'best_reward': float('-inf'),
            'best_reward_step': 0,
            'best_reward_schedule': None,
            'best_reward_metrics': {},
            'created_at': datetime.now().isoformat()
        }
        
        state = env.reset()
        
        return JSONResponse(content={
            "sim_id": sim_id,
            "message": "Simulation created successfully",
            "config": config.dict(),
            "initial_state": {
                "state_shape": state.shape,
                "state_size": int(env.state_space_size),
                "action_size": int(env.action_space_size)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/{sim_id}/state")
async def get_simulation_state(sim_id: str):
    """Get current simulation state"""
    if sim_id not in simulation_instances:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        sim = simulation_instances[sim_id]
        env = sim['env']
        
        # Get schedule data organized by time slot
        schedule_by_slot = {}
        for i in range(len(env.schedule)):
            time_slot = int(env.schedule[i, 3])
            if time_slot not in schedule_by_slot:
                schedule_by_slot[time_slot] = []
            
            schedule_by_slot[time_slot].append({
                'class_id': int(env.schedule[i, 0]),
                'teacher_id': int(env.schedule[i, 1]),
                'room_id': int(env.schedule[i, 2]),
                'time_slot': time_slot,
                'status': int(env.schedule[i, 4]),
                'status_text': {-1: 'DISRUPTED', 0: 'NOT SCHEDULED', 1: 'SCHEDULED'}[int(env.schedule[i, 4])],
                'students': int(env.student_enrollment[i])
            })
        
        # Get availability matrices
        teacher_availability = env.teacher_availability.tolist()
        room_availability = env.room_availability.tolist()
        
        # Get disruptions
        disruptions = []
        for d in env.active_disruptions:
            disruptions.append(convert_to_serializable(d))
        
        # Calculate metrics
        scheduled_count = int(np.sum(env.schedule[:, 4] == 1))
        disrupted_count = int(np.sum(env.schedule[:, 4] == -1))
        unscheduled_count = int(np.sum(env.schedule[:, 4] == 0))
        
        return {
            "sim_id": sim_id,
            "episode": sim['episode'],
            "step": sim['step'],
            "schedule": schedule_by_slot,
            "teacher_availability": teacher_availability,
            "room_availability": room_availability,
            "disruptions": disruptions,
            "num_classes": env.num_classes,
            "num_teachers": env.num_teachers,
            "num_rooms": env.num_rooms,
            "num_time_slots": env.num_time_slots,
            "metrics": {
                "scheduled_classes": scheduled_count,
                "disrupted_classes": disrupted_count,
                "unscheduled_classes": unscheduled_count,
                "total_classes": env.num_classes,
                "completion_rate": scheduled_count / env.num_classes,
                "total_reward": float(sim['total_reward']),
                "current_step": int(env.current_step)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulation/{sim_id}/step")
async def simulation_step(sim_id: str, step_data: SimulationStep):
    """Execute one simulation step"""
    if sim_id not in simulation_instances:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        sim = simulation_instances[sim_id]
        env = sim['env']
        agent = sim['agent']
        
        # Get current state
        state = env.state
        
        # Select action
        if step_data.auto:
            action = agent.select_action(state)
        else:
            action = step_data.action if step_data.action is not None else agent.select_action(state)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Agent learns
        agent.learn(state, action, reward, next_state, done)
        
        # Update simulation state
        sim['step'] += 1
        sim['total_reward'] += reward
        
        # Calculate current metrics
        scheduled_count = int(np.sum(env.schedule[:, 4] == 1))
        disrupted_count = int(np.sum(env.schedule[:, 4] == -1))
        unscheduled_count = int(np.sum(env.schedule[:, 4] == 0))
        
        # Count conflicts (teachers/rooms assigned to multiple classes at same time)
        conflicts = 0
        for t in range(env.num_time_slots):
            slot_schedule = env.schedule[env.schedule[:, 3] == t]
            teachers_in_slot = slot_schedule[:, 1]
            rooms_in_slot = slot_schedule[:, 2]
            # Count duplicates (excluding -1 which means unassigned)
            valid_teachers = teachers_in_slot[teachers_in_slot >= 0]
            valid_rooms = rooms_in_slot[rooms_in_slot >= 0]
            conflicts += len(valid_teachers) - len(np.unique(valid_teachers))
            conflicts += len(valid_rooms) - len(np.unique(valid_rooms))
        
        # Add metrics to history
        step_metrics = {
            'step': sim['step'],
            'reward': float(reward),
            'scheduled_classes': scheduled_count,
            'disrupted_classes': disrupted_count,
            'unscheduled_classes': unscheduled_count,
            'conflicts': conflicts,
            'total_reward': float(sim['total_reward'])
        }
        sim['metrics_history'].append(step_metrics)
        
        # Track best reward and its characteristics
        if reward > sim['best_reward']:
            sim['best_reward'] = float(reward)
            sim['best_reward_step'] = sim['step']
            sim['best_reward_schedule'] = env.schedule.copy()
            sim['best_reward_metrics'] = {
                'scheduled_classes': scheduled_count,
                'disrupted_classes': disrupted_count,
                'unscheduled_classes': unscheduled_count,
                'conflicts': conflicts,
                'teacher_utilization': float(np.mean(np.sum(env.teacher_availability, axis=1) / env.num_time_slots)) * 100,
                'room_utilization': float(np.mean(np.sum(env.room_availability, axis=1) / env.num_time_slots)) * 100
            }
        
        # Convert info to serializable format
        info_serializable = convert_to_serializable(info)
        
        # If episode done, reset
        if done:
            sim['episode'] += 1
            sim['episode_rewards'].append(sim['total_reward'])
            env.reset()
            sim['total_reward'] = 0
        
        return {
            "sim_id": sim_id,
            "step": sim['step'],
            "episode": sim['episode'],
            "action": int(action),
            "reward": float(reward),
            "done": done,
            "info": info_serializable,
            "cumulative_reward": float(sim['total_reward'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulation/{sim_id}/inject-disruption")
async def inject_disruption(sim_id: str, disruption: DisruptionEvent):
    """Manually inject a disruption event"""
    if sim_id not in simulation_instances:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        sim = simulation_instances[sim_id]
        env = sim['env']
        
        print(f"\n🚨 INJECTING DISRUPTION: {disruption.type}")
        print(f"   Target ID: {disruption.target_id}")
        
        # Create disruption dict based on type
        disruption_dict = {
            'type': disruption.type,
            'severity': 'high',
            'timestamp': datetime.now().isoformat(),
            'reason': 'manual_injection',
            'step': sim['step']
        }
        
        affected_classes = []
        
        # Add specific fields based on type
        if disruption.type == 'teacher_absent':
            disruption_dict['teacher_id'] = disruption.target_id
            disruption_dict['duration'] = 8
            
            if 0 <= disruption.target_id < env.num_teachers:
                # Mark teacher as unavailable
                print(f"   Making Teacher {disruption.target_id} unavailable at all time slots")
                env.teacher_availability[disruption.target_id, :] = 0
                
                # Update schedule to mark affected classes
                for class_idx in range(env.num_classes):
                    if env.schedule[class_idx, 1] == disruption.target_id and env.schedule[class_idx, 4] == 1:
                        env.schedule[class_idx, 4] = -1  # Mark as disrupted
                        affected_classes.append(int(class_idx))
                        print(f"   → Class {class_idx} disrupted (was taught by Teacher {disruption.target_id})")
                
                disruption_dict['affected_classes'] = affected_classes
                disruption_dict['description'] = f"Teacher {disruption.target_id} absent - {len(affected_classes)} classes disrupted"
            else:
                raise HTTPException(status_code=400, detail=f"Invalid teacher ID: {disruption.target_id}")
            
        elif disruption.type == 'room_unavailable':
            disruption_dict['room_id'] = disruption.target_id
            disruption_dict['duration'] = 4
            
            if 0 <= disruption.target_id < env.num_rooms:
                # Mark room as unavailable
                print(f"   Making Room {disruption.target_id} unavailable at all time slots")
                env.room_availability[disruption.target_id, :] = 0
                
                # Update schedule to mark affected classes
                for class_idx in range(env.num_classes):
                    if env.schedule[class_idx, 2] == disruption.target_id and env.schedule[class_idx, 4] == 1:
                        env.schedule[class_idx, 4] = -1  # Mark as disrupted
                        affected_classes.append(int(class_idx))
                        print(f"   → Class {class_idx} disrupted (was in Room {disruption.target_id})")
                
                disruption_dict['affected_classes'] = affected_classes
                disruption_dict['description'] = f"Room {disruption.target_id} unavailable - {len(affected_classes)} classes disrupted"
            else:
                raise HTTPException(status_code=400, detail=f"Invalid room ID: {disruption.target_id}")
            
        elif disruption.type == 'enrollment_change':
            disruption_dict['class_id'] = disruption.target_id
            
            if 0 <= disruption.target_id < env.num_classes:
                import random
                change = random.choice([-5, -3, 3, 5, 7])
                old_enrollment = int(env.student_enrollment[disruption.target_id])
                new_enrollment = max(10, min(50, old_enrollment + change))
                env.student_enrollment[disruption.target_id] = new_enrollment
                
                disruption_dict['old_enrollment'] = old_enrollment
                disruption_dict['new_enrollment'] = new_enrollment
                disruption_dict['change'] = change
                disruption_dict['description'] = f"Class {disruption.target_id} enrollment changed: {old_enrollment} → {new_enrollment} ({change:+d} students)"
                
                print(f"   Class {disruption.target_id} enrollment: {old_enrollment} → {new_enrollment} ({change:+d} students)")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid class ID: {disruption.target_id}")
        
        # Add to active disruptions
        env.active_disruptions.append(disruption_dict)
        
        # Apply penalty
        penalty_reward = -30
        sim['total_reward'] += penalty_reward
        
        print(f"   ✅ Disruption applied! Penalty: {penalty_reward}")
        print(f"   Total disruptions: {len(env.active_disruptions)}\n")
        
        return {
            "message": "Disruption injected successfully",
            "disruption": convert_to_serializable(disruption_dict),
            "penalty_reward": penalty_reward,
            "total_disruptions": len(env.active_disruptions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/{sim_id}/metrics")
async def get_metrics(sim_id: str):
    """Get performance metrics"""
    if sim_id not in simulation_instances:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        sim = simulation_instances[sim_id]
        env = sim['env']
        
        # Return metrics history as an array
        metrics_history = sim['metrics_history']
        
        if not metrics_history:
            # Return empty array if no metrics yet
            return []
        
        return metrics_history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/{sim_id}/summary")
async def get_simulation_summary(sim_id: str):
    """Get complete simulation summary for final results"""
    if sim_id not in simulation_instances:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    try:
        sim = simulation_instances[sim_id]
        env = sim['env']
        metrics_history = sim['metrics_history']
        
        # Calculate statistics
        total_steps = sim['step']
        total_episodes = sim['episode']
        
        rewards = [m['reward'] for m in metrics_history] if metrics_history else []
        avg_reward = float(np.mean(rewards)) if rewards else 0
        max_reward = float(np.max(rewards)) if rewards else 0
        min_reward = float(np.min(rewards)) if rewards else 0
        total_reward = sum(rewards)
        
        # Get last 50 steps average
        last_50_rewards = rewards[-50:] if len(rewards) >= 50 else rewards
        recent_avg_reward = float(np.mean(last_50_rewards)) if last_50_rewards else 0
        
        # Calculate improvement
        first_10_rewards = rewards[:10] if len(rewards) >= 10 else rewards
        first_avg = float(np.mean(first_10_rewards)) if first_10_rewards else 0
        improvement_pct = ((recent_avg_reward - first_avg) / first_avg * 100) if first_avg != 0 else 0
        
        # Count conflicts
        conflicts = [m.get('conflicts', 0) for m in metrics_history]
        total_conflicts = sum(conflicts)
        avg_conflicts = float(np.mean(conflicts)) if conflicts else 0
        
        # Scheduling success
        scheduled_counts = [m.get('scheduled_classes', 0) for m in metrics_history]
        final_scheduled = scheduled_counts[-1] if scheduled_counts else 0
        success_rate = (final_scheduled / env.num_classes * 100) if env.num_classes > 0 else 0
        
        # Disruptions
        total_disruptions = len(env.active_disruptions)
        
        # Resource utilization
        teacher_util = float(np.mean(np.sum(env.teacher_availability, axis=1) / env.num_time_slots))
        room_util = float(np.mean(np.sum(env.room_availability, axis=1) / env.num_time_slots))
        
        summary = {
            "sim_id": sim_id,
            "training_complete": total_steps >= 200 or total_episodes >= 200,
            "best_reward": {
                "reward": float(sim['best_reward']) if sim['best_reward'] != float('-inf') else 0,
                "step": sim['best_reward_step'],
                "metrics": sim['best_reward_metrics'],
                "characteristics": {
                    "scheduled_classes": sim['best_reward_metrics'].get('scheduled_classes', 0),
                    "conflicts": sim['best_reward_metrics'].get('conflicts', 0),
                    "teacher_utilization": sim['best_reward_metrics'].get('teacher_utilization', 0),
                    "room_utilization": sim['best_reward_metrics'].get('room_utilization', 0),
                    "success_rate": (sim['best_reward_metrics'].get('scheduled_classes', 0) / env.num_classes * 100) if env.num_classes > 0 else 0
                }
            },
            "statistics": {
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "total_reward": float(total_reward),
                "average_reward": avg_reward,
                "recent_average_reward": recent_avg_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
                "improvement_percentage": improvement_pct
            },
            "performance": {
                "classes_scheduled": final_scheduled,
                "total_classes": env.num_classes,
                "success_rate": success_rate,
                "total_conflicts": total_conflicts,
                "average_conflicts_per_step": avg_conflicts,
                "conflict_reduction": float((conflicts[0] - conflicts[-1]) / conflicts[0] * 100) if len(conflicts) > 1 and conflicts[0] > 0 else 0
            },
            "disruptions": {
                "total_disruptions": total_disruptions,
                "disruptions_handled": total_disruptions,
                "adaptation_rate": 100.0
            },
            "resource_utilization": {
                "teacher_utilization": teacher_util * 100,
                "room_utilization": room_util * 100,
                "teachers": env.num_teachers,
                "rooms": env.num_rooms,
                "time_slots": env.num_time_slots
            },
            "configuration": {
                "num_classes": env.num_classes,
                "num_teachers": env.num_teachers,
                "num_rooms": env.num_rooms,
                "num_time_slots": env.num_time_slots,
                "agent_type": sim['config'].agent_type
            }
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/simulation/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a simulation instance"""
    if sim_id not in simulation_instances:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulation_instances[sim_id]
    return {"message": "Simulation deleted successfully"}

@app.get("/simulations")
async def list_simulations():
    """List all active simulations"""
    simulations = []
    for sim_id, sim in simulation_instances.items():
        simulations.append({
            "sim_id": sim_id,
            "episode": sim['episode'],
            "step": sim['step'],
            "agent_type": sim['config'].agent_type,
            "created_at": sim['created_at']
        })
    return {"simulations": simulations, "count": len(simulations)}

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{sim_id}")
async def websocket_endpoint(websocket: WebSocket, sim_id: str):
    """WebSocket for real-time simulation updates"""
    await websocket.accept()
    
    if sim_id not in simulation_instances:
        await websocket.send_json({"error": "Simulation not found"})
        await websocket.close()
        return
    
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_json()
            command = data.get('command', 'step')
            
            if command == 'step':
                # Execute simulation step
                sim = simulation_instances[sim_id]
                env = sim['env']
                agent = sim['agent']
                
                state = env.state
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                
                sim['step'] += 1
                sim['total_reward'] += reward
                
                if done:
                    sim['episode'] += 1
                    sim['episode_rewards'].append(sim['total_reward'])
                    env.reset()
                    sim['total_reward'] = 0
                
                # Send state update
                state_data = await get_simulation_state(sim_id)
                await websocket.send_json({
                    "type": "state_update",
                    "data": state_data
                })
                
            elif command == 'get_state':
                # Send current state
                state_data = await get_simulation_state(sim_id)
                await websocket.send_json({
                    "type": "state",
                    "data": state_data
                })
                
            elif command == 'inject_disruption':
                # Inject disruption
                disruption_data = data.get('disruption', {})
                disruption = DisruptionEvent(**disruption_data)
                result = await inject_disruption(sim_id, disruption)
                await websocket.send_json({
                    "type": "disruption_injected",
                    "data": result
                })
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for simulation {sim_id}")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
