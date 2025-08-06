from pydantic import BaseModel, Field
from typing import Dict

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

class FinishUploadPayload(BaseModel):
    uploadId: str = Field(..., alias="uploadId")
    fileName: str = Field(..., alias="fileName")
    userId: str = Field(..., alias="userId")
    totalChunks: int = Field(..., alias="totalChunks")
