from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from entities.models import TaskResponse, StatusResponse
import time
import uvicorn
from typing import Optional

import os
import logging
from subprocess import Popen

import shutil
from transcriber import Transcriber
from auth import check_token
from config import SOURCE_FOLDER, RESULT_FOLDER
import sys


# from .transcribe_and_diarize import process_large_file
transcriber = Transcriber()

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, filename='_test.log', filemode='w')
logger = logging.getLogger(__name__)

transcriber = Transcriber()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def get_status() -> str:
    
    logger.info(f"Checking service health")

    return JSONResponse({'health': 'OK'})


@app.post("/start_transcribing", response_model=TaskResponse)
async def start_transcribing(file: UploadFile = File(),
                            diarize: bool = Query(False, description="Enable diarization"),
                            authenticated: bool = Depends(check_token)
                            ) -> TaskResponse:
                            
    start_time = time.time()
    logger.info(f"Starting transcription for file: {file.filename}, diarization: {diarize}")
    
    task_id = transcriber.create_task()
    file_path = os.path.join(SOURCE_FOLDER, f"{task_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        await transcriber.set_task_data(task_id, status='UPLOADING', upload_progress=100)
        
        c_env = os.environ.copy()

        args = [sys.executable, r'app/background_transcriber.py', task_id, file_path, str(diarize)]
        p = Popen(args, env=c_env)
        
        end_time = time.time()
        logger.info(f"Task {task_id} created and started processing in {end_time - start_time:.2f} seconds")
        return TaskResponse(task_id=task_id, message="Task processing started")
    except Exception as e:
        logger.error(f"Error in start_transcribing for task {task_id}: {str(e)}", exc_info=True)
        await transcriber.set_task_data(task_id, status='ERROR', error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_status", response_model=StatusResponse)
async def get_status(task_id: str, 
                    authenticated: bool = Depends(check_token)
                    ) -> StatusResponse:
    
    logger.info(f"Getting status for task: {task_id}")
    status = await transcriber.get_task_status(task_id)
    if status is None:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    logger.info(f"Status for task {task_id}: {status}")
    return status


@app.get("/get_file")
async def get_file(task_id: str, 
                   authenticated: bool = Depends(check_token)
                   ):
    logger.info(f"Getting file for task: {task_id}")
    status = await transcriber.get_task_status(task_id)
    if status is None:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    if status.status != 'READY':
        logger.warning(f"Task {task_id} is not ready. Current status: {status.status}")
        raise HTTPException(status_code=400, detail=f'Transcribing by task "{task_id}" is not ready')
    
    file_name = f"{task_id}_result.txt"
    file_path = os.path.join(RESULT_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        logger.error(f"Result file not found for task {task_id}")
        raise HTTPException(status_code=404, detail="Result file not found")
    
    logger.info(f"Returning file for task {task_id}: {file_path}")
    return FileResponse(path=file_path, filename=file_name, media_type='text/plain')


@app.get("/clear_results", description='Method for removing task and results text. You can remove all tasks or only ready tasks')
async def clear_results(authenticated: bool = Depends(check_token), only_ready: Optional[bool] = True):

    if only_ready is None:
        only_ready = True

    await transcriber.clear_results(only_ready)
    return JSONResponse({'message': 'results are cleared'})


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    # uvicorn.run(app, host="0.0.0.0", port=9040)
    uvicorn.run(app, host="localhost", port=9040)
