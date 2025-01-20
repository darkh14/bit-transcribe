import torch
import numpy as np
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from omegaconf import OmegaConf
import librosa
import soundfile as sf
import os
import json
import logging
import time
import traceback
import wget
from nemo.collections.asr.models import EncDecCTCModel
import shutil
from pydub import AudioSegment
from typing import List, Tuple
import pickle

CHUNK_SIZE = 10 * 60 * 1000  # 10 минут в миллисекундах

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ASR_MODEL_PATH = os.environ.get('ASR_MODEL_PATH', '/app/models/asr_model.nemo')
DIAR_MODEL_PATH = os.environ.get('DIAR_MODEL_PATH', '/app/models/diar_model.nemo')

import pprint

def log_embeddings_content(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info("Contents of subsegments_scale0_embeddings.pkl:")
        logger.info(data)
    except Exception as e:
        logger.error(f"Error reading embeddings file: {str(e)}")


def extract_speaker_embeddings(embeddings_file, cluster_label_file):
    try:
        # Загрузка эмбеддингов
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        embeddings = data['preprocessed_audio']
        
        # Загрузка меток кластеров (спикеров)
        with open(cluster_label_file, 'r') as f:
            cluster_labels = [line.strip().split()[-1] for line in f]
        
        # Проверка, что количество эмбеддингов соответствует количеству меток
        if len(embeddings) != len(cluster_labels):
            logger.warning(f"Mismatch between number of embeddings ({len(embeddings)}) and cluster labels ({len(cluster_labels)})")
        
        # Преобразование тензора PyTorch в numpy array, если необходимо
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        # Группировка эмбеддингов по спикерам
        speaker_embeddings = {}
        for emb, label in zip(embeddings, cluster_labels):
            if label not in speaker_embeddings:
                speaker_embeddings[label] = []
            speaker_embeddings[label].append(emb)
        
        # Вычисление среднего эмбеддинга для каждого спикера
        for speaker in speaker_embeddings:
            speaker_embeddings[speaker] = np.mean(speaker_embeddings[speaker], axis=0)
        
        logger.info(f"Extracted embeddings for {len(speaker_embeddings)} speakers")
        return speaker_embeddings
    except Exception as e:
        logger.error(f"Error extracting speaker embeddings: {str(e)}")
        return None

def check_file_content(file_path, file_type):
    logger.info(file_path)
    if os.path.exists(file_path):
        logger.info(f"Found {file_type} file: {file_path}")
        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.endswith('.label'):
                with open(file_path, 'r') as f:
                    data = f.read().splitlines()
            else:
                with open(file_path, 'r') as f:
                    data = f.read()
            
            logger.info(f"{file_type} data type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"Number of keys: {len(data)}")
                logger.info(f"Sample keys: {list(data.keys())[:5]}")
            elif isinstance(data, list):
                logger.info(f"Number of items: {len(data)}")
                logger.info(f"Sample items: {data[:5]}")
            else:
                logger.info(f"Data preview: {str(data)[:200]}")
        except Exception as e:
            logger.error(f"Error reading {file_type} file: {str(e)}")
    else:
        logger.warning(f"{file_type} file not found: {file_path}")

def log_directory_structure(start_path):
    """
    Рекурсивно выводит структуру директорий и файлов, начиная с start_path.
    """
    logger.info(f"Directory structure of {start_path}:")
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            logger.info(f"{sub_indent}{file}")

def split_audio(file_path: str) -> List[str]:
    """Разделяет большой аудиофайл на чанки по 10 минут."""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i, chunk in enumerate(audio[::CHUNK_SIZE]):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def process_chunk(chunk_path: str) -> Tuple[str, dict, dict, dict]:
    """Обрабатывает отдельный чанк аудио."""
    result, diar_hyp, speaker_embeddings, trans_info_dict = process_large_file_with_speaker_sync(chunk_path)
    return result, diar_hyp, speaker_embeddings, trans_info_dict

def synchronize_speakers(chunks_results: List[Tuple[str, dict, dict, dict]]) -> List[Tuple[str, dict, dict, dict]]:
    """Синхронизирует спикеров между чанками, используя эмбеддинги."""
    global_speakers = {}  # Глобальный словарь спикеров: {speaker_id: embedding}
    synchronized_results = []
    
    for i, (result, diar_hyp, speaker_embeddings, trans_info_dict) in enumerate(chunks_results):
        logger.info(f"Processing chunk {i}")
        
        local_to_global = {}
        for local_spk, local_emb in speaker_embeddings.items():
            if not global_speakers:
                # Для первого чанка просто добавляем всех спикеров
                global_spk = f"SPEAKER_{len(global_speakers)}"
                global_speakers[global_spk] = local_emb
                local_to_global[local_spk] = global_spk
            else:
                # Для последующих чанков ищем наиболее похожего спикера
                similarities = [cosine_similarity(local_emb, global_emb) for global_emb in global_speakers.values()]
                most_similar = max(enumerate(similarities), key=lambda x: x[1])
                if most_similar[1] > 0.85:  # Повышенный порог схожести
                    global_spk = list(global_speakers.keys())[most_similar[0]]
                    local_to_global[local_spk] = global_spk
                    # Обновляем глобальный эмбеддинг (среднее значение)
                    global_speakers[global_spk] = (global_speakers[global_spk] + local_emb) / 2
                else:
                    # Если нет достаточно похожего спикера, создаем нового
                    global_spk = f"SPEAKER_{len(global_speakers)}"
                    global_speakers[global_spk] = local_emb
                    local_to_global[local_spk] = global_spk
        
        # Обновляем метки спикеров в trans_info_dict
        updated_trans_info_dict = trans_info_dict.copy()
        for item in updated_trans_info_dict['preprocessed_audio']['words']:
            item['speaker'] = local_to_global[item['speaker']]
        for item in updated_trans_info_dict['preprocessed_audio']['sentences']:
            item['speaker'] = local_to_global[item['speaker']]
        
        # Обновляем diar_hyp
        updated_diar_hyp = {
            'preprocessed_audio': [
                f"{start} {end} {local_to_global[speaker]}" 
                for start, end, speaker in [segment.split() for segment in diar_hyp['preprocessed_audio']]
            ]
        }
        
        synchronized_results.append((result, updated_diar_hyp, speaker_embeddings, updated_trans_info_dict))
        
        logger.info(f"Chunk {i} speaker mapping: {local_to_global}")
    
    logger.info(f"Total unique speakers after synchronization: {len(global_speakers)}")
    return synchronized_results
def preprocess_audio(file_path, target_sr=16000):
    logger.info(f"Preprocessing audio file: {file_path}")
    try:
        audio, sr = librosa.load(file_path, sr=None)
        logger.info(f"Original sample rate: {sr}")
        if sr != target_sr:
            logger.info(f"Resampling from {sr} to {target_sr}")
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        audio = librosa.util.normalize(audio)
        preprocessed_path = 'preprocessed_audio.wav'
        sf.write(preprocessed_path, audio, target_sr)
        logger.info(f"Preprocessed audio saved to {preprocessed_path}")
        return preprocessed_path
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise



def process_large_file(file_path: str) -> dict:
    logger.info("start splitting")
    chunks = split_audio(file_path)
    logger.info(f"chunks received: {len(chunks)}")
    chunks_results = [process_chunk(chunk) for chunk in chunks]
    logger.info(f"chunks processed: {len(chunks_results)}")
    synchronized_results = synchronize_speakers(chunks_results)
    logger.info(f"synchronization completed: {len(synchronized_results)} results")

    logger.info("start results combining")

    # Объединяем результаты
    final_result = {
        "transcript": "",
        "words": [],
        "sentences": [],
        "speaker_count": 0
    }

    unique_speakers = set()

    for i, (_, _, _, trans_info) in enumerate(synchronized_results):
        # Обновляем транскрипцию
        chunk_transcript = ""
        for sentence in trans_info['preprocessed_audio']['sentences']:
            chunk_transcript += f"[{sentence['speaker']}]: {sentence['text']} "
        final_result["transcript"] += chunk_transcript
        
        # Обрабатываем слова
        for word in trans_info['preprocessed_audio']['words']:
            word['start_time'] += i * CHUNK_SIZE / 1000
            word['end_time'] += i * CHUNK_SIZE / 1000
            final_result["words"].append(word)
            unique_speakers.add(word['speaker'])
        
        # Обрабатываем предложения
        for sentence in trans_info['preprocessed_audio']['sentences']:
            sentence['start_time'] = float(sentence['start_time']) + i * CHUNK_SIZE / 1000
            sentence['end_time'] = float(sentence['end_time']) + i * CHUNK_SIZE / 1000
            final_result["sentences"].append(sentence)
            unique_speakers.add(sentence['speaker'])

    final_result["transcript"] = final_result["transcript"].strip()
    final_result["speaker_count"] = len(unique_speakers)

    logger.info("Results combined")
    logger.info(f"Transcript length: {len(final_result['transcript'])}")
    logger.info(f"Total words: {len(final_result['words'])}")
    logger.info(f"Total sentences: {len(final_result['sentences'])}")
    logger.info(f"Speaker count: {final_result['speaker_count']}")
    logger.info(f"Unique speakers: {unique_speakers}")

    # Очистка временных файлов
    for chunk in chunks:
        os.remove(chunk)
    logger.info("Temporary files removed")

    return final_result


def process_large_file_with_speaker_sync(file_path):
    start_time = time.time()
    logger.info(f"Starting processing of file: {file_path}")
    
    try:
        preprocessed_file = preprocess_audio(file_path)
        
        # Download the configuration file
        DOMAIN_TYPE = "meeting"
        CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        
        if not os.path.exists(CONFIG_FILE_NAME):
            wget.download(CONFIG_URL)
        
        # Load and modify the configuration
        cfg = OmegaConf.load(CONFIG_FILE_NAME)
        cfg.diarizer.manifest_filepath = 'input_manifest.json'
        cfg.diarizer.out_dir = './temp_diar_output'
        cfg.diarizer.speaker_embeddings.model_path = DIAR_MODEL_PATH
        cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
        cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
        cfg.diarizer.clustering.parameters.oracle_num_speakers = False
        cfg.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'
        cfg.diarizer.asr.model_path = 'stt_ru_conformer_ctc_large' #ASR_MODEL_PATH
        cfg.diarizer.asr.parameters.asr_based_vad = True
        cfg.diarizer.asr.parameters.threshold = 100
        cfg.diarizer.speaker_embeddings.parameters.save_embeddings = True

        # Create manifest for the input file
        manifest_content = {
            'audio_filepath': preprocessed_file, 
            'offset': 0, 
            'duration': None, 
            'label': 'infer', 
            'text': '-', 
            'num_speakers': None, 
            'rttm_filepath': None, 
            'uem_filepath': None
        }
        with open('input_manifest.json', 'w') as fp:
            json.dump(manifest_content, fp)
            fp.write('\n')

        # Initialize ASR model
        asr_model = EncDecCTCModel.restore_from(ASR_MODEL_PATH)
        
        # Initialize ASRDecoderTimeStamps with the configuration
        asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
        asr_model = asr_decoder_ts.set_asr_model()

        # Run ASR
        word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
        logger.info(f"ASR completed. word_hyp: {word_hyp}, word_ts_hyp: {word_ts_hyp}")

        # Initialize OfflineDiarWithASR
        asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
        asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset

        # Perform diarization
        diar_hyp, speaker_embeddings = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
        if diar_hyp is None:
            logger.error("Diarization failed: diar_hyp is None")
            return "Diarization failed", None, None
        logger.info(f"Diarization completed. diar_hyp: {diar_hyp}")

        # Get transcript with speaker labels
        trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
        if trans_info_dict is None:
            logger.error("Failed to get transcript with speaker labels")
            return "Failed to get transcript with speaker labels", diar_hyp, speaker_embeddings
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        logger.info(f"diar_hyp: {diar_hyp}")
        logger.info(f"word_hyp: {word_hyp}")
        logger.info(f"word_ts_hyp: {word_ts_hyp}")
        logger.info(f"trans_info_dict: {trans_info_dict}")
        # Read the text transcription file
        transcription_path = f"./temp_diar_output/pred_rttms/{os.path.basename(preprocessed_file).split('.')[0]}.txt"
        try:
            with open(transcription_path, 'r') as f:
                transcript = f.read().splitlines()
        except FileNotFoundError:
            logger.error(f"Transcription file not found: {transcription_path}")
            transcript = ["Transcription file not found"]


        # После выполнения диаризации:
        embeddings_dir = "./temp_diar_output/embeddings/"
        if not os.path.exists(embeddings_dir):
            logger.warning(f"Embeddings directory does not exist: {embeddings_dir}")
            try:
                os.makedirs(embeddings_dir)
                logger.info(f"Created embeddings directory: {embeddings_dir}")
            except Exception as e:
                logger.error(f"Failed to create embeddings directory: {str(e)}")

        embeddings_file = f"{embeddings_dir}{os.path.basename(preprocessed_file).split('.')[0]}.npy"
        logger.info(f"Looking for embeddings file: {embeddings_file}")

        # Проверим содержимое директории embeddings
        embeddings_dir = "./temp_diar_output/embeddings/"
        if os.path.exists(embeddings_dir):
            logger.info(f"Contents of {embeddings_dir}:")
            for file in os.listdir(embeddings_dir):
                logger.info(f"  - {file}")
                check_file_content(os.path.join(embeddings_dir, file), "Embeddings file")
        else:
            logger.warning(f"Embeddings directory not found: {embeddings_dir}")

        if os.path.exists(embeddings_file):
            try:
                speaker_embeddings = np.load(embeddings_file, allow_pickle=True).item()
                logger.info(f"Speaker embeddings loaded. Number of speakers: {len(speaker_embeddings)}")
            except Exception as e:
                logger.error(f"Error loading speaker embeddings: {str(e)}")
                speaker_embeddings = None
        else:
            logger.warning(f"Speaker embeddings file not found: {embeddings_file}")
            speaker_embeddings = None

        # Проверим содержимое директории с эмбеддингами
        logger.info(f"Contents of {embeddings_dir}:")
        for file in os.listdir(embeddings_dir):
            logger.info(f"  - {file}")


        # Read the JSON transcription file
        json_path = f"./temp_diar_output/pred_rttms/{os.path.basename(preprocessed_file).split('.')[0]}.json"
        try:
            with open(json_path, 'r') as f:
                json_content = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_path}")
            json_content = {"error": "JSON file not found"}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {json_path}")
            json_content = {"error": "Invalid JSON"}

        logger.info("Conversation transcript:")
        for line in transcript:
            logger.info(line)

        logger.info("Combining results")
        #result = []
        #for line in transcript:
        #    result.append(line)
        #result.append("\nDetailed JSON output:")
        #result.append(json.dumps(json_content, indent=2))

        
        end_time = time.time()
        logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")


        # В функции process_large_file_with_speaker_sync:
        embeddings_file = "./temp_diar_output/speaker_outputs/embeddings/subsegments_scale0_embeddings.pkl"
        cluster_label_file = "./temp_diar_output/speaker_outputs/subsegments_scale0_cluster.label"
        speaker_embeddings = extract_speaker_embeddings(embeddings_file, cluster_label_file)
        logger.info(f"SPEAKER EMBEDDINGS: {speaker_embeddings}")
        if speaker_embeddings:
            logger.info(f"Successfully extracted embeddings for {len(speaker_embeddings)} speakers")
            for speaker, emb in speaker_embeddings.items():
                logger.info(f"Speaker {speaker} embedding shape: {emb.shape}")
        else:
            logger.warning("Failed to extract speaker embeddings")

        # Формируем результат в том же формате, что и раньше
        result = []
        for segment in diar_hyp['preprocessed_audio']:
            parts = segment.split()
            if len(parts) >= 3:
                start, end, speaker = parts[:3]
                text = ' '.join(parts[3:]) if len(parts) > 3 else ''
                result.append(f"[{start} - {end}] {speaker}: {text}")

        logger.info("Results generated")
        logger.info(f"Number of segments: {len(result)}")

        logger.info("Removing temporary files")
        if os.path.exists(preprocessed_file):
            os.remove(preprocessed_file)
        if os.path.exists('input_manifest.json'):
            os.remove('input_manifest.json')
        if os.path.exists('./temp_diar_output'):
            shutil.rmtree('./temp_diar_output')

        # Возвращаем speaker_embeddings вместе с другими результатами
        # Возвращаем результат в прежнем формате, но добавляем trans_info_dict
        return "\n".join(result), diar_hyp, speaker_embeddings, trans_info_dict

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error during processing: {str(e)}", None, None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Дополнительные вспомогательные функции могут быть добавлены здесь при необходимости

if __name__ == "__main__":
    # Этот блок может быть использован для локального тестирования
    test_file = "path/to/test/audio/file.wav"
    result = process_large_file_with_speaker_sync(test_file)
    print(result)
