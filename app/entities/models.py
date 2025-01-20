from pydantic import BaseModel
from typing import Optional

class TaskResponse(BaseModel):
    task_id: str
    message: str

class StatusResponse(BaseModel):
    status: str
    progress: int
    upload_progress: int
    transcribe_progress: int
    diarization_progress: int  # diarization
    current_batch: Optional[int] = None
    total_batches: Optional[int] = None
    # current_diarization_chunk: Optional[int] = None  # diarization
    # total_diarization_chunks: Optional[int] = None  # diarization
    total_transcription_time: Optional[float] = None
    total_diarization_time: Optional[float] = None  # diarization
    error: Optional[str] = None
