import sys
import logging
import os
import asyncio

from transcriber import Transcriber
from utils.asr.models import GigaAMModel

logging.basicConfig(level=logging.INFO, filename='_test.log', filemode='w')
logger = logging.getLogger(__name__)


async def transcribe(task_id, file_path, diarize):
    try:
        transcriber = Transcriber()
        model = GigaAMModel(diarize=diarize)

        transcriber.set_model(model)

        transcriber.read_task(task_id)
        await transcriber.set_task_data(task_id, status='UPLOADING', upload_progress=100)
        await transcriber.process_file(task_id, file_path, diarize)
    except Exception as e:
        logger.error(f"Error in start_transcribing for task {task_id}: {str(e)}", exc_info=True)
        await transcriber.set_task_data(task_id, status='ERROR')

async def main():
    _, task_id, file_path, diarize = sys.argv
    diarize = True if diarize == 'True' else False
    print(task_id, file_path, diarize)
    await transcribe(task_id, file_path, diarize)


if len(sys.argv) == 4:

    asyncio.run(main())    

