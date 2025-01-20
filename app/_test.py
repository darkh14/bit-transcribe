# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from entities.models import TaskResponse, StatusResponse
# import time
# import uvicorn

# import os
# import logging

# import shutil
# from transcriber import Transcriber
# from utils.asr.models import GigaAMCTCDiarizeModel
# from auth import check_token
# from config import SOURCE_FOLDER, RESULT_FOLDER

# from uuid import uuid4
# import asyncio
import sys

from os import environ
import os


from subprocess import Popen

# logging.basicConfig(level=logging.INFO, filename='_test.log', filemode='w')

# async def main():

#     # initial_file_path = 'Запись встречи 18.09.2024 15-43-49 - запись.webm'
#     initial_file_path = 'GMT20241007-105742_Recording_2560x1080.mp4'
#     initial_file_path = os.path.join('test_data', initial_file_path)
#     filename = os.path.basename(initial_file_path)

#     transcriber = Transcriber()
#     model = GigaAMCTCDiarizeModel()
#     transcriber.set_model(model)
#     task_id = transcriber.create_task()

#     file_path = os.path.join(SOURCE_FOLDER, f"{task_id}_{filename}")
#     shutil.copyfile(initial_file_path, file_path)

#     await transcriber.set_task_data(task_id, status='UPLOADING', upload_progress=100)

#     await transcriber.process_file(task_id, file_path, diarize=True)


# asyncio.run(main())

# cmd = 'python utils/asr/va_converter.py'
        # args = shlex.split(cmd)
# print(sys.executable)
myenv = environ.copy()
# if 'VIRTUAL_ENV' in environ:
#     myenv['PATH'] = ':'.join(
#         [x for x in environ['PATH'].split(':')
#             if x != os.path.join(environ['VIRTUAL_ENV'], 'bin')])


args = [sys.executable, os.path.abspath(r'app/utils/va_converter.py')]
p = Popen(args, env=myenv)

# print(p)
        #['/bin/vikings',
        # '-input',
        # 'eggs.txt',
        # '-output',
        # 'spam spam.txt',
        # '-cmd',
        # "echo '$MONEY'"]



        # p = Process(target=self.__class__.convert_video_to_audio_proc, args=(self.video_file_path, self.audio_file_path))
        # p.start()
        # #p.join()
        # time.sleep(30)