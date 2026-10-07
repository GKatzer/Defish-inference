# core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Network
    vds1_ip: str
    vds1_port: int = 8000
    ml_service_port: int = 8000
    
    # Paths
    model_path: str = "./models"
    
    # Models
    ensemble_models: List[str] = ["best.onnx"]
    ensemble_weights: List[float] = [1.0]
    
    # Performance
    workers: int = 2
    batch_size: int = 4
    max_concurrent_requests: int = 10
    
    # Timeouts
    request_timeout: int = 30

    # Rate limit per client IP (in memory of each process): at most N requests per window of S seconds
    rate_limit_requests: int = 60
    rate_limit_window: int = 60

    # Classifier speed knobs, passed to load_classifier() by main.py. The defaults are the behaviour the
    # recorded accuracy figures were made with; on a slow CPU (a small VDS) they are the first thing to try.
    #   classifier_tta: also embed the mirror image of every crop (two embedder passes per crop instead of one)
    #   classifier_chunk_size: crops per embedder call; 0 = all crops in one call (needs memory for 2N images)
    classifier_tta: bool = True
    classifier_chunk_size: int = 1

    @property
    def vds1_url(self) -> str:
        return f"http://{self.vds1_ip}:{self.vds1_port}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields

settings = Settings()

# In the code use:
# from core.config import settings
# print(settings.vds1_url)  # http://192.168.1.100:8001