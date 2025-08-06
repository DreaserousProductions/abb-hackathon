import logging
from fastapi import APIRouter, HTTPException
from app.models.csv_models import TrainModelRequest, TrainModelResponse
from app.services import csv_service

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/csv",
    tags=["CSV Training"]
)

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
