from typing import Optional
from pydantic import BaseModel

class AttackParameters(BaseModel):
    model: str
    datasetStructure: str
    csvPath: Optional[str]
    datasetSize: int
    numClasses: int
    batchSize: int
    numRestarts: int
    stepSize: float
    maxIterations: int
    callbackInterval: int
    ptFilePath: Optional[str]
    zipFilePath: Optional[str]
    budget: int

class AttackStatistics(BaseModel):
    MSE: float = 0
    PSNR: float = 0
    SSIM: float = 0

class AttackProgress(BaseModel):
    message_type: str = "AttackProgress"
    current_iteration: int = 0
    max_iterations: int = 0
    current_restart: int = 0
    max_restarts: int = 0
    current_batch: int = 0
    max_batches: int = 0
    time_taken: float = 0
    statistics: AttackStatistics = AttackStatistics()
    reconstructed_image: Optional[str] = None # base64 encoded image
