from typing import Optional
import asyncio
import logging
import os
import uuid
import time
import json
from entities.models import StatusResponse
from utils.asr.models import GigaAMModel
from config import SOURCE_FOLDER, AUDIO_FOLDER, RESULT_FOLDER, TEMP_RESULT_FOLDER, TASKS_FOLDER

import shutil

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self):
        self.tasks = {}
        self.pid = os.getpid()
        self.model: Optional[GigaAMModel]
        # self.coder = GigaAMCoder()
        # self.model = GigaAMRNNTModel(self.coder)
        # self.diarization_pipeline = None
        # self.diarization_chunk_duration = 300  # 5 minutes for diarization chunks

    def create_task(self):
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            'pid': self.pid,
            'status': 'CREATED',
            'progress': 0,
            'upload_progress': 0,
            'transcribe_progress': 0,
            'diarization_progress': 0,
            'current_batch': None,
            'total_batches': None,
            # 'current_diarization_chunk': None,
            # 'total_diarization_chunks': None,
            'start_time': None,
            'total_transcription_time': None,
            'total_diarization_time': None,
            'error': None
        }
        self.write_task(task_id, self.tasks[task_id])
        logger.info(f"Created new task with ID: {task_id}")
        return task_id
    
    def write_task(self, task_id, task):
        
        with open(os.path.join(TASKS_FOLDER, '{}_.json'.format(task_id)), 'w') as fp:
            json.dump(task, fp)
            fp.flush()

    def read_task(self, task_id):
        result = None
        
        filename = os.path.join(TASKS_FOLDER, '{}_.json'.format(task_id))
        
        if os.path.exists(filename):
            with open(filename, 'r') as fp:
                result = json.load(fp)

        self.tasks[task_id] = result

        return result

    def set_model(self, model):
        self.model = model
        self.model.status_writer = self

    async def set_task_data(self, task_id, **kwargs):
        
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            if kwargs.get('status') == 'PROCESSING' and self.tasks[task_id]['start_time'] is None:
                self.tasks[task_id]['start_time'] = time.time()
            elif kwargs.get('status') == 'READY':
                if self.tasks[task_id]['start_time']:
                    self.tasks[task_id]['total_transcription_time'] = time.time() - self.tasks[task_id]['start_time']
            logger.info(f"Updated task {task_id} status: {self.tasks[task_id]}")

            self.write_task(task_id, self.tasks[task_id])

    async def get_task_status(self, task_id):

        # status_data = None
        # if task_id in self.tasks:
        #     status_data = self.tasks[task_id].copy()
        #     logger.info(f"Retrieved status for task {task_id}: {status_data}")
        
        # if not status_data:
        status_data = self.read_task(task_id)
        
        if status_data:
            if 'start_time' in status_data:
                del status_data['start_time']
            return StatusResponse(**status_data)
        
        logger.warning(f"Attempted to get status for non-existent task: {task_id}")
        return None

    async def process_file(self, task_id, file_path, diarize=False):
        if not self.model:
            raise ValueError('Model is not set. Set model before process')
        start_time = time.time()
        logger.info(f"Starting processing for task {task_id}")
        try:
            await self.set_task_data(task_id, status='PROCESSING', progress=0)
            result_data = await self.model.process(task_id, file_path)
            self.model = None
            await self._write_result_file(task_id, result_data, diarize=diarize)
            
            end_time = time.time()
            logger.info(f"Task {task_id} completed successfully in {end_time - start_time:.2f} seconds")
            await self.set_task_data(task_id, status='READY', progress=100, transcribe_progress=100, diarization_progress=100 if diarize else 0)
        except Exception as e:
            logger.error(f"Error in task processing for {task_id}: {str(e)}", exc_info=True)
            await self.set_task_data(task_id, status='ERROR', error=str(e))
        finally:
            await self._cleanup_files(task_id)

    async def _write_result_file(self, task_id, transcription_result, diarize):
        file_name = f"{task_id}_result.txt"
        file_path = os.path.join(RESULT_FOLDER, file_name)
    
        logger.info(f"Writing result file with diarization for task {task_id}")
        # async with asyncio.Lock():

        if diarize:
            with open(file_path, 'w', encoding='utf-8') as f:
                current_speaker = None
                current_text = []
                current_start = None
            
                for result_line in transcription_result:
                    speaker, start, end, text = result_line['speaker'], result_line['start_time'], result_line['end_time'], result_line['text']
                    if not text:
                        continue
                    if speaker != current_speaker:
                        if current_speaker is not None:
                            f.write(f"[{self._format_time(current_start)} - {self._format_time(end)}] {current_speaker}: {' '.join(current_text)}\n")
                        current_speaker = speaker
                        current_text = [text]
                        current_start = start
                    else:
                        current_text.append(text)
            
                if current_speaker is not None:
                    f.write(f"[{self._format_time(current_start)} - {self._format_time(end)}] {current_speaker}: {' '.join(current_text)}\n")
        else: 
            # if isinstance(transcription_result[0], tuple):  # This means we have hypotheses
            #     transcriptions, boundaries, _ = transcription_result
            # else:
            transcriptions, boundaries = transcription_result
            transcriptions = transcriptions[0]
            
            file_name = f"{task_id}_result.txt"
            file_path = os.path.join(RESULT_FOLDER, file_name)
            
            logger.info(f"Writing result file for task {task_id}")
            async with asyncio.Lock():
                with open(file_path, 'w', encoding='utf-8') as f:
                    for transcription, boundary in zip(transcriptions, boundaries):
                        f.write(f"[{self._format_time(boundary[0])} - {self._format_time(boundary[1])}]: {transcription}\n")
            logger.info(f"Result file written for task {task_id}: {file_path}")
        
            logger.info(f"Result file with diarization written for task {task_id}: {file_path}")

    async def clear_results(self, only_ready=True):
        
        logger.info('Clearing results. Start')
        for file in os.listdir(TASKS_FOLDER):
            task_id = os.path.splitext(file)[0][:-1]
            self.read_task(task_id)

        tasks_not_ready = [key for key, value in self.tasks.items() if value['status'] != 'READY']

        folders = [RESULT_FOLDER, TASKS_FOLDER, SOURCE_FOLDER, AUDIO_FOLDER, TEMP_RESULT_FOLDER]

        # results
        for folder in folders:
            for file in os.listdir(folder):
                task_id = os.path.splitext(file)[0].split('_')[0]

                if only_ready and task_id in tasks_not_ready:
                    continue
                
                file_path = os.path.join(folder, file)
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info('Clearing results. File "{}" was removed'.format(file_path))
                    else:
                        shutil.rmtree(file_path) 
                        logger.info('Clearing results. Directory "{}" was removed with all content'.format(file_path)) 

        logger.info('Clearing results. Done')                                                

    # async def _write_result_file(self, task_id, result_data):
    #     if isinstance(result_data[0], tuple):  # This means we have hypotheses
    #         transcriptions, boundaries, _ = result_data
    #     else:
    #         transcriptions, boundaries = result_data
        
    #     file_name = f"{task_id}_result.txt"
    #     file_path = os.path.join(RESULT_FOLDER, file_name)
        
    #     logger.info(f"Writing result file for task {task_id}")
    #     async with asyncio.Lock():
    #         with open(file_path, 'w', encoding='utf-8') as f:
    #             for transcription, boundary in zip(transcriptions, boundaries):
    #                 f.write(f"[{self._format_time(boundary[0])} - {self._format_time(boundary[1])}]: {transcription}\n")
    #     logger.info(f"Result file written for task {task_id}: {file_path}")

    async def _cleanup_files(self, task_id):
        logger.info(f"Starting cleanup for task {task_id}")
        source_path = os.path.join(SOURCE_FOLDER, f"{task_id}_*")
        audio_path = os.path.join(AUDIO_FOLDER, f"{task_id}_*")
        
        for path in [source_path, audio_path]:
            for file in os.listdir(os.path.dirname(path)):
                if file.startswith(f"{task_id}_"):
                    os.remove(os.path.join(os.path.dirname(path), file))
                    logger.info(f"Removed file: {file}")
        for folder in os.listdir(TEMP_RESULT_FOLDER):
            subfolder_path = os.path.join(TEMP_RESULT_FOLDER, folder)
            if os.path.isdir(subfolder_path) and folder.startswith(f"{task_id}_"):
                shutil.rmtree(subfolder_path)

        logger.info(f"Cleanup completed for task {task_id}")

    @staticmethod
    def _format_time(seconds):
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
