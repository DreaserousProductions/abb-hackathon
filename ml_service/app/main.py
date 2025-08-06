# app/main.py

from fastapi import FastAPI
from app.routers import csv_processor

# --- App Initialization ---
app = FastAPI(
    title="Modular CSV Processor API",
    description="An API to upload and process CSV files, now with a modular structure.",
    version="2.0.0"
)

# --- Include Routers ---
# Include the router from the csv_processor module.
# Any new routers for different functionalities can be added here.
app.include_router(csv_processor.router)


# --- Root Endpoint ---
@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Modular CSV Processor API. Visit /docs for documentation."}
