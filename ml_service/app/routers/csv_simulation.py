import asyncio
import logging
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services import csv_service

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/csv",
    tags=["CSV Simulation"]
)

class SimulationConnectionManager:
    """Manages active WebSocket connections and their stop signals."""
    
    def __init__(self):
        self.active_connections: Dict[str, asyncio.Event] = {}

    async def connect(self, simulation_id: str):
        self.active_connections[simulation_id] = asyncio.Event()

    def disconnect(self, simulation_id: str):
        if simulation_id in self.active_connections:
            del self.active_connections[simulation_id]
            log.info(f"Manager cleaned up connection for {simulation_id}")

    def signal_stop(self, simulation_id: str):
        """Sets the stop event, signaling the simulation loop to terminate."""
        if simulation_id in self.active_connections:
            self.active_connections[simulation_id].set()
            log.info(f"Stop signal sent for {simulation_id}")

    def is_stop_signaled(self, simulation_id: str) -> bool:
        """Checks if the stop event has been set."""
        if simulation_id in self.active_connections:
            return self.active_connections[simulation_id].is_set()
        return True

manager = SimulationConnectionManager()

async def stream_simulation_results(websocket: WebSocket, user_id: str, dataset_id: str):
    """Helper task that streams results from the service generator to the websocket."""
    simulation_id = f"{user_id}:{dataset_id}"
    try:
        async for result in csv_service.run_simulation_for_websocket(user_id, dataset_id):
            if manager.is_stop_signaled(simulation_id):
                log.info(f"[{simulation_id}] Halting stream due to stop signal.")
                await websocket.send_json({"status": "stopped"})
                break

            await websocket.send_json(result)

        if not manager.is_stop_signaled(simulation_id):
            await websocket.send_json({"status": "finished"})

    except Exception as e:
        log.error(f"Error in simulation stream for {simulation_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": f"An unexpected error occurred during simulation: {e}"})
        except RuntimeError:
            pass
    finally:
        log.info(f"[{simulation_id}] Result streaming task finished.")

@router.websocket("/simulation-ws")
async def simulation_websocket_endpoint(websocket: WebSocket):
    """
    Handles WebSocket connections for live simulation.
    - Listens for a "start" message to begin streaming.
    - Listens for a "stop" message to halt the stream.
    """
    simulation_id = ""
    simulation_task = None
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            log.info(f"WebSocket received message: {data}")

            action = data.get("action")
            user_id = data.get("userId")
            dataset_id = data.get("datasetId")

            if not user_id or not dataset_id:
                await websocket.send_json({"error": "userId and datasetId are required."})
                continue

            simulation_id = f"{user_id}:{dataset_id}"

            if action == "start":
                if simulation_task and not simulation_task.done():
                    log.warning(f"[{simulation_id}] Simulation already running. Ignoring start command.")
                    continue

                await manager.connect(simulation_id)
                simulation_task = asyncio.create_task(
                    stream_simulation_results(websocket, user_id, dataset_id)
                )

            elif action == "stop":
                log.info(f"[{simulation_id}] Received 'stop' action from client.")
                manager.signal_stop(simulation_id)
                await websocket.send_json({"status": "stopping"})

    except WebSocketDisconnect:
        log.info(f"Client disconnected from simulation {simulation_id}. Cleaning up.")
    except Exception as e:
        log.error(f"An unexpected error occurred in the websocket endpoint: {e}", exc_info=True)
    finally:
        if simulation_id:
            manager.signal_stop(simulation_id)
            manager.disconnect(simulation_id)
        log.info(f"WebSocket for {simulation_id} closed.")
