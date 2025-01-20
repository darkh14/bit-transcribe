import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_FOLDER = os.path.join(BASE_DIR, 'data', 'source')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'data', 'audio')
RESULT_FOLDER = os.path.join(BASE_DIR, 'data', 'result')
CONFIG_FOLDER = os.path.join(BASE_DIR, 'model_configs')
TEMP_RESULT_FOLDER = os.path.join(BASE_DIR, 'tmp')
MODEL_DATA_FOLDER = os.path.join(BASE_DIR, 'model_data')
TASKS_FOLDER = os.path.join(BASE_DIR, 'tasks')

for folder in [SOURCE_FOLDER, AUDIO_FOLDER, RESULT_FOLDER, CONFIG_FOLDER, TEMP_RESULT_FOLDER, TASKS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

HF_TOKEN = os.getenv('HF_TOKEN')
# RNNT_CONFIG_PATH = os.getenv('RNNT_CONFIG_PATH', '../models_data/models--gigaam--rnnt/rnnt_model_config.yaml')
# RNNT_WEIGHTS_PATH = os.getenv('RNNT_WEIGHTS_PATH', '../models_data/models--gigaam--rnnt/rnnt_model_weights.ckpt')
# AUTH_SERVICE_URL = os.getenv('AUTH_SERVICE_URL', 'https://devtools.1bitai.ru/9010')
