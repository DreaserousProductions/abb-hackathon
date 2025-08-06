import logging
import os
import shutil
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.models.csv_models import FinishUploadPayload
from app.services import csv_service

# Directory setup
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

CHUNK_UPLOAD_DIR = "temp_chunks"
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/csv",
    tags=["CSV Upload"]
)

@router.post("/upload-chunk", summary="Upload a single file chunk")
async def upload_chunk_route(
    file: UploadFile = File(...),
    uploadId: str = Form(..., alias="uploadId"),
    chunkIndex: int = Form(..., alias="chunkIndex"),
    userId: str = Form(..., alias="userId")
):
    """Receives a single chunk of a large file and saves it to a temporary directory."""
    try:
        chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, uploadId)
        os.makedirs(chunk_dir, exist_ok=True)
        
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunkIndex}")
        log.info(f"Receiving chunk {chunkIndex} for upload {uploadId} to {chunk_path}")

        with open(chunk_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"status": "chunk uploaded", "chunkIndex": chunkIndex}

    except Exception as e:
        log.error(f"Error saving chunk {chunkIndex} for upload {uploadId}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save chunk {chunkIndex}.")

@router.post("/finish-upload", summary="Reassemble chunks and process the complete file")
async def finish_upload_route(payload: FinishUploadPayload):
    upload_id = payload.uploadId
    chunk_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
    
    if not os.path.isdir(chunk_dir):
        raise HTTPException(status_code=404, detail="Upload ID not found.")

    reassembled_path = os.path.join(TEMP_UPLOAD_DIR, f"{upload_id}_{payload.fileName}")

    try:
        log.info(f"Reassembling {payload.totalChunks} chunks for upload ID {upload_id}...")
        
        with open(reassembled_path, "wb", buffering=1024*1024) as final_file:
            for i in range(payload.totalChunks):
                chunk_path = os.path.join(chunk_dir, f"chunk_{i}")
                
                if not os.path.exists(chunk_path):
                    raise HTTPException(status_code=400, detail=f"Upload failed: Missing chunk {i}.")
                
                with open(chunk_path, "rb", buffering=1024*1024) as chunk_file:
                    shutil.copyfileobj(chunk_file, final_file, length=1024*1024)

        log.info(f"File reassembled at {reassembled_path}. Starting processing...")

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
        if os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir)
            log.info(f"Cleaned up chunk directory: {chunk_dir}")
