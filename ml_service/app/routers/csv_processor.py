# app/routers/csv_processor.py

import asyncio
import logging
import os
import shutil  # Import shutil for cleanup and optimized copying
from typing import Dict
from fastapi import (APIRouter, Body, File, Form, HTTPException, Request,
                    UploadFile, WebSocket, WebSocketDisconnect)
from pydantic import BaseModel, Field

# Assuming csv_service has a function that processes a file from a path
from app.services import csv_service

# --- Directory Setup ---
# Directory to store the final reassembled file before processing
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Directory to store the temporary chunks for each upload
CHUNK_UPLOAD_DIR = "temp_chunks"
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Create a new router object
router = APIRouter(
    prefix="/csv",
    tags=["CSV Processor"]
)

# --- Pydantic Models ---
class DateRange(BaseModel):
    start: str
    end: str

class ValidateRangesRequest(BaseModel):
    userId: str
    datasetId: str
    dateRanges: Dict[str, DateRange]

class TrainModelRequest(BaseModel):
    userId: str
    datasetId: str
    dateRanges: Dict[str, DateRange]

class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1Score: float = Field(..., alias="f1Score")
    trueNegative: int = Field(..., alias="trueNegative")
    falsePositive: int = Field(..., alias="falsePositive")
    falseNegative: int = Field(..., alias="falseNegative")
    truePositive: int = Field(..., alias="truePositive")

class Plots(BaseModel):
    featureImportance: str = Field(..., alias="featureImportance")
    trainingPlot: str = Field(..., alias="trainingPlot")

class TrainModelResponse(BaseModel):
    metrics: Metrics
    plots: Plots

# --- NEW: Model for the finish-upload request body ---
class FinishUploadPayload(BaseModel):
    uploadId: str = Field(..., alias="uploadId")
    fileName: str = Field(..., alias="fileName")
    userId: str = Field(..., alias="userId")
    totalChunks: int = Field(..., alias="totalChunks")

# --- Endpoint to receive individual file chunks ---
@router.post("/upload-chunk", summary="Upload a single file chunk")
async def upload_chunk_route(
    file: UploadFile = File(...),
    uploadId: str = Form(..., alias="uploadId"),
    chunkIndex: int = Form(..., alias="chunkIndex"),
    userId: str = Form(..., alias="userId")
):
    """
    Receives a single chunk of a large file and saves it to a temporary directory.
    """
    try:
        chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, uploadId)
        os.makedirs(chunk_dir, exist_ok=True)
        
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunkIndex}")
        log.info(f"Receiving chunk {chunkIndex} for upload {uploadId} to {chunk_path}")

        # Asynchronously read the chunk and write to disk
        with open(chunk_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"status": "chunk uploaded", "chunkIndex": chunkIndex}

    except Exception as e:
        log.error(f"Error saving chunk {chunkIndex} for upload {uploadId}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save chunk {chunkIndex}.")

# --- OPTIMIZED: Endpoint to finalize the upload and start processing ---
@router.post("/finish-upload", summary="Reassemble chunks and process the complete file")
async def finish_upload_route(payload: FinishUploadPayload):
    upload_id = payload.uploadId
    chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
    
    if not os.path.isdir(chunk_dir):
        raise HTTPException(status_code=404, detail="Upload ID not found.")

    reassembled_path = os.path.join(TEMP_UPLOAD_DIR, f"{upload_id}_{payload.fileName}")

    try:
        log.info(f"Reassembling {payload.totalChunks} chunks for upload ID {upload_id}...")
        
        # OPTIMIZATION: Use buffered file operations with larger buffer size for better I/O performance
        with open(reassembled_path, "wb", buffering=1024*1024) as final_file:  # 1MB buffer
            for i in range(payload.totalChunks):
                chunk_path = os.path.join(chunk_dir, f"chunk_{i}")
                
                if not os.path.exists(chunk_path):
                    raise HTTPException(status_code=400, detail=f"Upload failed: Missing chunk {i}.")
                
                with open(chunk_path, "rb", buffering=1024*1024) as chunk_file:  # 1MB buffer
                    # CRITICAL OPTIMIZATION: Use shutil.copyfileobj with large buffer for maximum I/O efficiency
                    shutil.copyfileobj(chunk_file, final_file, length=1024*1024)  # 1MB copy buffer

        log.info(f"File reassembled at {reassembled_path}. Starting processing...")

        # Process the reassembled file - this now uses the optimized single-pass approach
        results = csv_service.process_csv_to_parquet(
            temp_csv_path=reassembled_path,
            user_id=payload.userId
        )

        log.info(f"Successfully processed reassembled file for upload {upload_id}.")
        return results

    except Exception as e:
        log.error(f"Error during file reassembly or processing for {upload_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during file processing.")
    
    finally:
        # Clean up chunk directory
        if os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir)
            log.info(f"Cleaned up chunk directory: {chunk_dir}")

@router.post("/validate-ranges", summary="Validate date ranges and get record counts")
async def validate_ranges_route(request: ValidateRangesRequest):
    """
    Receives date ranges, validates them against the stored dataset, and returns counts.
    """
    log.info("--- Received request for /validate-ranges ---")
    log.info(f"Parsed request object: userId='{request.userId}', datasetId='{request.datasetId}'")

    try:
        ranges_dict = request.dict().get("dateRanges", {})
        if not ranges_dict:
            raise HTTPException(status_code=400, detail="dateRanges dictionary is missing or empty.")

        log.info(f"Extracted ranges dictionary: {ranges_dict}")
        log.info("Calling csv_service.validate_and_count_ranges...")

        results = csv_service.validate_and_count_ranges(
            user_id=request.userId,
            dataset_id=request.datasetId,
            ranges=ranges_dict
        )

        log.info(f"Service call successful. Result status: {results.get('status')}")

    except ValueError as e:
        log.error(f"ValueError in /validate-ranges for user '{request.userId}': {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Unhandled exception in /validate-ranges for user '{request.userId}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

    # Handle validation results
    if results.get("status") == "Invalid":
        detail = results.get("detail", "Invalid ranges provided.")
        log.warning(f"Validation failed for user '{request.userId}': {detail}")
        raise HTTPException(status_code=400, detail=detail)

    log.info("--- Request for /validate-ranges completed successfully ---")
    return results

@router.post("/train", summary="Train a model and get metrics and plots", response_model=TrainModelResponse)
async def train_model_route(request: TrainModelRequest):
    """
    Trains an XGBoost model and returns a detailed analysis including:
    - **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score.
    - **Visual Plots**: A Base64-encoded feature importance plot.
    """
    log.info(f"--- Received request for /train for user '{request.userId}' ---")

    try:
        ranges_dict = request.dict().get("dateRanges", {})
        if not all(k in ranges_dict for k in ['training', 'testing', 'simulation']):
            raise HTTPException(status_code=400, detail="Request must include 'training', 'testing', and 'simulation' date ranges.")

        print(ranges_dict)

        results = csv_service.train_model_with_ranges(
            user_id=request.userId,
            dataset_id=request.datasetId,
            ranges=ranges_dict
        )

        return results

    except ValueError as e:
        log.error(f"ValueError during training for user '{request.userId}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Unhandled exception during training for user '{request.userId}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

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
        return True  # Default to stopped if connection not found

manager = SimulationConnectionManager()

async def stream_simulation_results(websocket: WebSocket, user_id: str, dataset_id: str):
    """
    Helper task that streams results from the service generator to the websocket.
    """
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
            pass  # Websocket might already be closed
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

# # app/routers/csv_processor.py

# import asyncio
# import logging
# import os
# import shutil # Import shutil for cleanup
# from typing import Dict

# from fastapi import (APIRouter, Body, File, Form, HTTPException, Request,
#                      UploadFile, WebSocket, WebSocketDisconnect)
# from pydantic import BaseModel, Field

# # Assuming csv_service has a function that processes a file from a path
# from app.services import csv_service 

# # --- Directory Setup ---
# # Directory to store the final reassembled file before processing
# TEMP_UPLOAD_DIR = "temp_uploads"
# os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# # Directory to store the temporary chunks for each upload
# CHUNK_UPLOAD_DIR = "temp_chunks"
# os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)

# # Create a new router object
# router = APIRouter(
#     prefix="/csv",
#     tags=["CSV Processor"]
# )

# # --- Pydantic Models ---

# class DateRange(BaseModel):
#     start: str
#     end: str

# class ValidateRangesRequest(BaseModel):
#     userId: str
#     datasetId: str
#     dateRanges: Dict[str, DateRange]

# class TrainModelRequest(BaseModel):
#     userId: str
#     datasetId: str
#     dateRanges: Dict[str, DateRange]
    
# class Metrics(BaseModel):
#     accuracy: float
#     precision: float
#     recall: float
#     f1Score: float = Field(..., alias="f1Score")
#     trueNegative: int = Field(..., alias="trueNegative")
#     falsePositive: int = Field(..., alias="falsePositive")
#     falseNegative: int = Field(..., alias="falseNegative")
#     truePositive: int = Field(..., alias="truePositive")

# class Plots(BaseModel):
#     featureImportance: str = Field(..., alias="featureImportance")

# class TrainModelResponse(BaseModel):
#     metrics: Metrics
#     plots: Plots

# # --- NEW: Model for the finish-upload request body ---
# class FinishUploadPayload(BaseModel):
#     uploadId: str = Field(..., alias="uploadId")
#     fileName: str = Field(..., alias="fileName")
#     userId: str = Field(..., alias="userId")
#     totalChunks: int = Field(..., alias="totalChunks")


# # --- REMOVED: The old /process route is gone ---
# # @router.post("/process", ...) is no longer here.


# # --- NEW: Endpoint to receive individual file chunks ---
# @router.post("/upload-chunk", summary="Upload a single file chunk")
# async def upload_chunk_route(
#     file: UploadFile = File(...),
#     uploadId: str = Form(..., alias="uploadId"),
#     chunkIndex: int = Form(..., alias="chunkIndex"),
#     userId: str = Form(..., alias="userId") # Included for logging/potential validation
# ):
#     """
#     Receives a single chunk of a large file and saves it to a temporary directory.
#     """
#     try:
#         chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, uploadId)
#         os.makedirs(chunk_dir, exist_ok=True)
        
#         chunk_path = os.path.join(chunk_dir, f"chunk_{chunkIndex}")
#         log.info(f"Receiving chunk {chunkIndex} for upload {uploadId} to {chunk_path}")
        
#         # Asynchronously read the chunk and write to disk
#         with open(chunk_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)
            
#         return {"status": "chunk uploaded", "chunkIndex": chunkIndex}
#     except Exception as e:
#         log.error(f"Error saving chunk {chunkIndex} for upload {uploadId}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Could not save chunk {chunkIndex}.")

# # --- Endpoint to receive individual file chunks (Unchanged) ---
# @router.post("/upload-chunk", summary="Upload a single file chunk")
# async def upload_chunk_route(
#     file: UploadFile = File(...),
#     uploadId: str = Form(..., alias="uploadId"),
#     chunkIndex: int = Form(..., alias="chunkIndex"),
#     userId: str = Form(..., alias="userId")
# ):
#     try:
#         chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, uploadId)
#         os.makedirs(chunk_dir, exist_ok=True)
#         chunk_path = os.path.join(chunk_dir, f"chunk_{chunkIndex}")
#         log.info(f"Receiving chunk {chunkIndex} for upload {uploadId} to {chunk_path}")
#         with open(chunk_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)
#         return {"status": "chunk uploaded", "chunkIndex": chunkIndex}
#     except Exception as e:
#         log.error(f"Error saving chunk {chunkIndex} for upload {uploadId}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Could not save chunk {chunkIndex}.")


# # --- Endpoint to finalize the upload and start processing (Optimized Reassembly) ---
# @router.post("/finish-upload", summary="Reassemble chunks and process the complete file")
# async def finish_upload_route(payload: FinishUploadPayload):
#     upload_id = payload.uploadId
#     chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
    
#     if not os.path.isdir(chunk_dir):
#         raise HTTPException(status_code=404, detail="Upload ID not found.")

#     reassembled_path = os.path.join(TEMP_UPLOAD_DIR, f"{upload_id}_{payload.fileName}")

#     try:
#         log.info(f"Reassembling {payload.totalChunks} chunks for upload ID {upload_id}...")
#         with open(reassembled_path, "wb") as final_file:
#             for i in range(payload.totalChunks):
#                 chunk_path = os.path.join(chunk_dir, f"chunk_{i}")
#                 if not os.path.exists(chunk_path):
#                     raise HTTPException(status_code=400, detail=f"Upload failed: Missing chunk {i}.")
                
#                 with open(chunk_path, "rb") as chunk_file:
#                     # OPTIMIZATION: Use shutil.copyfileobj for efficient, buffered copying
#                     shutil.copyfileobj(chunk_file, final_file)
        
#         log.info(f"File reassembled at {reassembled_path}. Starting processing...")
#         results = csv_service.process_csv_to_parquet(
#             temp_csv_path=reassembled_path,
#             user_id=payload.userId
#         )
#         log.info(f"Successfully processed reassembled file for upload {upload_id}.")
#         return results
#     except Exception as e:
#         log.error(f"Error during file reassembly or processing for {upload_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An internal error occurred during file processing.")
#     finally:
#         if os.path.isdir(chunk_dir):
#             shutil.rmtree(chunk_dir)
#             log.info(f"Cleaned up chunk directory: {chunk_dir}")

# @router.post("/validate-ranges", summary="Validate date ranges and get record counts")
# async def validate_ranges_route(request: ValidateRangesRequest): # Removed raw_request as it's not needed
#     """
#     Receives date ranges, validates them against the stored dataset, and returns counts.
#     """
#     log.info("--- Received request for /validate-ranges ---")
#     log.info(f"Parsed request object: userId='{request.userId}', datasetId='{request.datasetId}'")
    
#     # The try...except block now *only* covers the parts that can raise UNEXPECTED errors.
#     try:
#         ranges_dict = request.dict().get("dateRanges", {})
#         if not ranges_dict:
#             # This is a bad request, not a server error. Raise it immediately.
#             raise HTTPException(status_code=400, detail="dateRanges dictionary is missing or empty.")

#         log.info(f"Extracted ranges dictionary: {ranges_dict}")
#         log.info("Calling csv_service.validate_and_count_ranges...")
        
#         # This service call is the main part that could fail unexpectedly (e.g., Redis down)
#         results = csv_service.validate_and_count_ranges(
#             user_id=request.userId,
#             dataset_id=request.datasetId,
#             ranges=ranges_dict
#         )
#         log.info(f"Service call successful. Result status: {results.get('status')}")

#     except ValueError as e:
#         # This catches expected errors like "Dataset not found" from the service
#         log.error(f"ValueError in /validate-ranges for user '{request.userId}': {e}", exc_info=True)
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         # This now only catches TRULY unexpected errors
#         log.error(f"Unhandled exception in /validate-ranges for user '{request.userId}': {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An internal server error occurred.")

#     # --- CORRECTED LOGIC ---
#     # This logic now runs AFTER the try...except block has successfully completed.
#     # It handles the EXPECTED business logic outcomes.
    
#     if results.get("status") == "Invalid":
#         # This is an expected validation failure. We raise a 400 error.
#         # Because this is outside the `try` block, it will NOT be caught by `except Exception`.
#         # FastAPI will correctly handle this and send a 400 response to the client.
#         detail = results.get("detail", "Invalid ranges provided.")
#         log.warning(f"Validation failed for user '{request.userId}': {detail}")
#         raise HTTPException(status_code=400, detail=detail)
        
#     log.info("--- Request for /validate-ranges completed successfully ---")
#     # If the status was "Valid", we return the successful results with a 200 OK status.
#     return results

# @router.post("/train", summary="Train a model and get metrics and plots", response_model=TrainModelResponse)
# async def train_model_route(request: TrainModelRequest):
#     """
#     Trains an XGBoost model and returns a detailed analysis including:
#     - **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score.
#     - **Visual Plots**: A Base64-encoded feature importance plot.
#     """
#     log.info(f"--- Received request for /train for user '{request.userId}' ---")
#     try:
#         ranges_dict = request.dict().get("dateRanges", {})
#         if not all(k in ranges_dict for k in ['training', 'testing', 'simulation']):
#             raise HTTPException(status_code=400, detail="Request must include 'training', 'testing', and 'simulation' date ranges.")
#         print(ranges_dict)
#         results = csv_service.train_model_with_ranges(
#             user_id=request.userId,
#             dataset_id=request.datasetId,
#             ranges=ranges_dict
#         )
#         return results
        
#     except ValueError as e:
#         log.error(f"ValueError during training for user '{request.userId}': {e}", exc_info=True)
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         log.error(f"Unhandled exception during training for user '{request.userId}': {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# class SimulationConnectionManager:
#     """Manages active WebSocket connections and their stop signals."""
#     def __init__(self):
#         self.active_connections: Dict[str, asyncio.Event] = {}

#     async def connect(self, simulation_id: str):
#         # Create a new event for this simulation. An event is like a flag.
#         self.active_connections[simulation_id] = asyncio.Event()

#     def disconnect(self, simulation_id: str):
#         if simulation_id in self.active_connections:
#             del self.active_connections[simulation_id]
#             log.info(f"Manager cleaned up connection for {simulation_id}")

#     def signal_stop(self, simulation_id: str):
#         """Sets the stop event, signaling the simulation loop to terminate."""
#         if simulation_id in self.active_connections:
#             self.active_connections[simulation_id].set()
#             log.info(f"Stop signal sent for {simulation_id}")

#     def is_stop_signaled(self, simulation_id: str) -> bool:
#         """Checks if the stop event has been set."""
#         if simulation_id in self.active_connections:
#             return self.active_connections[simulation_id].is_set()
#         return True # Default to stopped if connection not found

# manager = SimulationConnectionManager()


# async def stream_simulation_results(websocket: WebSocket, user_id: str, dataset_id: str):
#     """
#     Helper task that streams results from the service generator to the websocket.
#     """
#     simulation_id = f"{user_id}:{dataset_id}"
#     try:
#         # The 'async for' loop works perfectly with our async generator function
#         async for result in csv_service.run_simulation_for_websocket(user_id, dataset_id):
#             if manager.is_stop_signaled(simulation_id):
#                 log.info(f"[{simulation_id}] Halting stream due to stop signal.")
#                 await websocket.send_json({"status": "stopped"})
#                 break
            
#             await websocket.send_json(result)
        
#         # If the loop finishes without being stopped
#         if not manager.is_stop_signaled(simulation_id):
#             await websocket.send_json({"status": "finished"})

#     except Exception as e:
#         log.error(f"Error in simulation stream for {simulation_id}: {e}", exc_info=True)
#         try:
#             await websocket.send_json({"error": f"An unexpected error occurred during simulation: {e}"})
#         except RuntimeError:
#             pass # Websocket might already be closed
#     finally:
#         log.info(f"[{simulation_id}] Result streaming task finished.")


# @router.websocket("/simulation-ws")
# async def simulation_websocket_endpoint(websocket: WebSocket):
#     """
#     Handles WebSocket connections for live simulation.
#     - Listens for a "start" message to begin streaming.
#     - Listens for a "stop" message to halt the stream.
#     """
#     simulation_id = ""
#     simulation_task = None
#     await websocket.accept()
    
#     try:
#         while True:
#             data = await websocket.receive_json()
#             log.info(f"WebSocket received message: {data}")
#             action = data.get("action")
#             user_id = data.get("userId")
#             dataset_id = data.get("datasetId")

#             if not user_id or not dataset_id:
#                 await websocket.send_json({"error": "userId and datasetId are required."})
#                 continue
            
#             simulation_id = f"{user_id}:{dataset_id}"

#             if action == "start":
#                 if simulation_task and not simulation_task.done():
#                     log.warning(f"[{simulation_id}] Simulation already running. Ignoring start command.")
#                     continue
                
#                 await manager.connect(simulation_id)
#                 # Create and run the streaming task in the background
#                 simulation_task = asyncio.create_task(
#                     stream_simulation_results(websocket, user_id, dataset_id)
#                 )

#             elif action == "stop":
#                 log.info(f"[{simulation_id}] Received 'stop' action from client.")
#                 manager.signal_stop(simulation_id)
#                 await websocket.send_json({"status": "stopping"})

#     except WebSocketDisconnect:
#         log.info(f"Client disconnected from simulation {simulation_id}. Cleaning up.")
#     except Exception as e:
#         log.error(f"An unexpected error occurred in the websocket endpoint: {e}", exc_info=True)
#     finally:
#         if simulation_id:
#             # Ensure the simulation and connection are cleaned up
#             manager.signal_stop(simulation_id)
#             manager.disconnect(simulation_id)
#         log.info(f"WebSocket for {simulation_id} closed.")