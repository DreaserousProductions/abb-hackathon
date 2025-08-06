import logging
from fastapi import APIRouter, HTTPException
from app.models.csv_models import ValidateRangesRequest
from app.services import csv_service

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/csv",
    tags=["CSV Validation"]
)

@router.post("/validate-ranges", summary="Validate date ranges and get record counts")
async def validate_ranges_route(request: ValidateRangesRequest):
    """Receives date ranges, validates them against the stored dataset, and returns counts."""
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

    if results.get("status") == "Invalid":
        detail = results.get("detail", "Invalid ranges provided.")
        log.warning(f"Validation failed for user '{request.userId}': {detail}")
        raise HTTPException(status_code=400, detail=detail)

    log.info("--- Request for /validate-ranges completed successfully ---")
    return results
