from abc import ABCMeta, abstractmethod

from datetime import datetime
import sys
from typing import List, Dict, Type, Tuple
import os
from multiprocessing import Process

import numpy as np
import wget
import logging
import json
import shutil
from subprocess import Popen
import shlex

import torch
import torchaudio
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeaturesTA as NeMoFilterbankFeaturesTA
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps, if_none_get_default, PYCTCDECODE, WER_TS, get_uniqname_from_filepath
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE, EncDecRNNTBPEModel

from pyannote.audio import Pipeline
from pydub import AudioSegment
from io import BytesIO

from moviepy import VideoFileClip

from omegaconf import OmegaConf
from pymediainfo import MediaInfo
from config import RESULT_FOLDER, AUDIO_FOLDER, CONFIG_FOLDER, TEMP_RESULT_FOLDER, MODEL_DATA_FOLDER


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


class GigaDiarWithASR(OfflineDiarWithASR):

    def print_sentences(self, sentences: List[Dict[str, float]]):
        """
        Print a transcript with speaker labels and timestamps.

        Args:
            sentences (list):
                List containing sentence-level dictionaries.

        Returns:
            string_out (str):
                String variable containing transcript and the corresponding speaker label.
        """
        # init output
        string_out = ''

        for sentence in sentences:
            # extract info
            speaker = sentence['speaker']
            start_point = sentence['start_time']
            end_point = sentence['end_time']
            text = sentence['text']

            if self.params['colored_text']:
                color = self.color_palette.get(speaker, '\033[0;37m')
            else:
                color = ''

            # cast timestamp to the correct format
            datetime_offset = 0 #!!!!!!!!!!!
            if float(start_point) > 3600:
                time_str = '%H:%M:%S.%f'
            else:
                time_str = '%M:%S.%f'
            start_point, end_point = max(float(start_point), 0), max(float(end_point), 0)

            # print(start_point, datetime_offset, time_str)

            start_point_str = datetime.fromtimestamp(start_point - datetime_offset).strftime(time_str)[:-4]
            end_point_str = datetime.fromtimestamp(end_point - datetime_offset).strftime(time_str)[:-4]

            if self.params['print_time']:
                time_str = f'[{start_point_str} - {end_point_str}] '
            else:
                time_str = ''

            # string out concatenation
            string_out += f'{color}{time_str}{speaker}: {text}\n'

        return string_out
    

class GigaASRDecoderTimeStamps(ASRDecoderTimeStamps):

    def __init__(self, cfg_diarizer, audio_file_path, status_writer=None):
        super().__init__(cfg_diarizer)
        self.audio_file_path = audio_file_path
        self.status_writer = None

    async def set_task_data(self, task_id, **kwargs):
        if self.status_writer:
            await self.status_writer.set_task_data(task_id, **kwargs)

    def set_asr_model(self):
        """
        Initialize the parameters for the given ASR model.
        Currently, the following NGC models are supported:

            stt_en_quartznet15x5,
            stt_en_citrinet*,
            stt_en_conformer_ctc*

        To assign a proper decoding function for generating timestamp output,
        the name of .nemo file should include the architecture name such as:
        'quartznet', 'conformer', and 'citrinet'.

        decoder_delay_in_sec is the amount of delay that is compensated during the word timestamp extraction.
        word_ts_anchor_offset is the reference point for a word and used for matching the word with diarization labels.
        Each ASR model has a different optimal decoder delay and word timestamp anchor offset.
        To obtain an optimized diarization result with ASR, decoder_delay_in_sec and word_ts_anchor_offset
        need to be searched on a development set.
        """
        if 'quartznet' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_QuartzNet_CTC
            self.encdec_class = EncDecCTCModel
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.04)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.12)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 4)
            self.model_stride_in_secs = 0.02

        elif 'conformer' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_BPE_CTC
            self.encdec_class = EncDecCTCModelBPE
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.08)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.12)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 16)
            self.model_stride_in_secs = 0.04
            # Conformer requires buffered inference and the parameters for buffered processing.
            self.chunk_len_in_sec = 5
            self.total_buffer_in_secs = 25

        elif 'citrinet' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_CitriNet_CTC
            self.encdec_class = EncDecCTCModelBPE
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.16)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.2)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 4)
            self.model_stride_in_secs = 0.08
        elif 'ctc_gigaam' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_Gigaam_CTC
            self.encdec_class = EncDecCTCModel
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.04)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.12)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 4)
            self.model_stride_in_secs = 0.02            

        else:
            raise ValueError(f"Cannot find the ASR model class for: {self.params['self.ASR_model_name']}")

        if self.ASR_model_name.endswith('.nemo'):
            asr_model = self.encdec_class.restore_from(restore_path=self.ASR_model_name)
        else:
            asr_model = self.encdec_class.from_pretrained(model_name=self.ASR_model_name, strict=False)

        if self.ctc_decoder_params['pretrained_language_model']:
            if not PYCTCDECODE:
                raise ImportError(
                    'LM for beam search decoding is provided but pyctcdecode is not installed. Install pyctcdecode using PyPI: pip install pyctcdecode'
                )
            self.beam_search_decoder = self.load_LM_for_CTC_decoder(asr_model)
        else:
            self.beam_search_decoder = None

        asr_model.eval()
        return asr_model
    
    def run_ASR_Gigaam_CTC(self, asr_model: Type[EncDecCTCModel]) -> Tuple[Dict, Dict]:
        """
        Launch QuartzNet ASR model and collect logit, timestamps and text output.

        Args:
            asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_dict (dict):
                Dictionary containing the sequence of words from hypothesis.
            word_ts_dict (dict):
                Dictionary containing the time-stamps of words.
        """
        words_dict, word_ts_dict = {}, {}

        wer_ts = WER_TS(
            vocabulary=asr_model.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=asr_model._cfg.get("log_prediction", False),
        )

        with torch.amp.autocast(device_type='cuda'):

            # Initialize pyannote pipeline
            pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", 
            # use_auth_token=HF_TOKEN, 
            cache_dir=MODEL_DATA_FOLDER)
            pipeline = pipeline.to(DEVICE)


            # transcript_hyps_list = asr_model.transcribe(
            #     self.audio_file_list, batch_size=self.asr_batch_size, return_hypotheses=True
            # )  

            transcript_logits_list = []

            # for audio_file in self.audio_file_list:
                # print('-'*100)
            segments, boundaries = segment_audio(self.audio_file_path, pipeline)
            # transcript_hyps_list = asr_model.transcribe(
            #     segments, batch_size=self.asr_batch_size, return_hypotheses=True
            # ) 
            del pipeline

            transcript_hyps_list = asr_model.transcribe(
                segments, batch_size=2, return_hypotheses=True
            )             

            c_transcript_logits_list = [hyp.alignments for hyp in transcript_hyps_list]

            transcript_logits_list.append(torch.cat(c_transcript_logits_list, dim=0))

                # print(str(transcript_logits_list[0].shape), str(transcript_logits_list[1].shape), str(res.shape))

            for idx, logit_np in enumerate(transcript_logits_list):
                logit_np = logit_np.cpu().numpy()
                uniq_id = get_uniqname_from_filepath(self.audio_file_list[idx])
                if self.beam_search_decoder:
                    logging.info(
                        f"Running beam-search decoder on {uniq_id} with LM {self.ctc_decoder_params['pretrained_language_model']}"
                    )
                    hyp_words, word_ts = self.run_pyctcdecode(logit_np)
                else:
                    log_prob = torch.from_numpy(logit_np)
                    logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
                    greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
                    text, char_ts = wer_ts.ctc_decoder_predictions_tensor_with_ts(
                        greedy_predictions, predictions_len=logits_len
                    )
                    trans, char_ts_in_feature_frame_idx = self.clean_trans_and_TS(text[0], char_ts[0])
                    spaces_in_sec, hyp_words = self._get_spaces(
                        trans, char_ts_in_feature_frame_idx, self.model_stride_in_secs
                    )
                    word_ts = self.get_word_ts_from_spaces(
                        char_ts_in_feature_frame_idx, spaces_in_sec, end_stamp=logit_np.shape[0]
                    )
                word_ts = self.align_decoder_delay(word_ts, self.decoder_delay_in_sec)
                assert len(hyp_words) == len(word_ts), "Words and word timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts

        return words_dict, word_ts_dict
    



class AbstractModel:
    __metaclass__ = ABCMeta
    @abstractmethod
    def process(self, file_path,  progress_writer=None, params=None):
        ...


class GigaAMModel(AbstractModel):

    def __init__(self, diarize=True):

        self.source_file_path = ''
        self.audio_file_path = ''
        self.audio_file_no_ext = ''
        self.video_file_path = ''

        self.diarize = diarize

        self.status_writer = None
 
        self.is_video = False
        self.result_file_path = ''

        self.temp_result_folder = ''

        self.asr_model_path = os.path.join(MODEL_DATA_FOLDER, 'ctc_gigaam.nemo')

        self.inference_config_file = os.path.join(CONFIG_FOLDER, 'config_ctc.yaml') # 'diar_infer_meeting.yaml')
        self.input_manifest_file = os.path.join(CONFIG_FOLDER, 'input_manifest.json')
        self._set_config()
        self._set_asr_model()

    def _set_source_file_path(self, file_path):
        self.source_file_path = file_path

        self._check_set_file_type(self.source_file_path)
        if self.is_video:
            self.video_file_path = self.source_file_path
        
        self._set_audio_file_path()

        self.audio_file_name_no_ext = os.path.splitext(os.path.basename(self.audio_file_path))[0] 
        self.predicted_rttm_file = f'{self.audio_file_path}.rttm'

        self._set_manifest()
        
    def _check_set_file_type(self, file_path):
        fileInfo = MediaInfo.parse(file_path)
        error = True
        track_types = []
        for track in fileInfo.tracks:
            if track.track_type in ["Video", "Audio"]:
                error = False
                if track.track_type  == "Video":
                    self.is_video = True

            track_types.append(track.track_type)
        
        if error:
            raise ValueError('File type error! File "{}" has type "{}" is not media (video or audio)'.format(self.source_file_path, track_types))

    def _set_config(self):
        
        need_to_form_config = False
        if not os.path.exists(self.inference_config_file):
            need_to_form_config = True
            diar_model_name = 'diar_infer_meeting.yaml'
            config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{diar_model_name}"
            wget.download(config_url, self.inference_config_file)

        self.config = OmegaConf.load(self.inference_config_file)    

        if need_to_form_config:
            self.config.device = DEVICE.type

            self.config.diarizer.clustering.parameters.oracle_num_speakers = False

            self.config.diarizer.manifest_filepath = self.input_manifest_file
            # config.diarizer.msdd_model.model_path = 'diar_infer_meeting' # 'diar_msdd_telephonic'

            self.config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]
            self.config.diarizer.oracle_vad = False

            self.config.diarizer.out_dir = self.temp_result_folder

            self.config.diarizer.speaker_embeddings.model_path = 'titanet_large'
            self.config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
            self.config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.1]
            self.config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
            self.config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
            self.config.diarizer.vad.parameters.offset = 0.6
            self.config.diarizer.vad.parameters.onset = 0.8
            self.config.diarizer.vad.parameters.pad_offset = -0.05

            self.config.num_workers = 0

            self.config.diarizer.asr.model_path = self.asr_model_path
            self.config.diarizer.asr.batch_size = 10

            OmegaConf.save(self.config, self.inference_config_file)

        return self.config

    def _set_manifest(self):

        audio_file_name_no_ext = os.path.splitext(os.path.basename(self.audio_file_path))[0]
        
        meta = {
            'audio_filepath': self.audio_file_path,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': 5,
            'rttm_filepath': f'{self.temp_result_folder}/pred_rttms/{audio_file_name_no_ext}.rttm',
            'uem_filepath': None
            }

        with open(self.input_manifest_file, 'w') as fp:
            json.dump(meta, fp)
            fp.write('\n')
        
        return meta
    
    async def _convert_video_to_audio(self):
        logger.info('Converting. Start') 
        clip = VideoFileClip(self.video_file_path)
        clip.audio.write_audiofile(self.audio_file_path, fps=16000, ffmpeg_params=["-ac", "1"])
        clip.close()
        logger.info('Converting. Done') 

    def _copy_source_to_audio_folder(self):
        shutil.copyfile(self.source_file_path, self.audio_file_path)

    def _set_audio_file_path(self):
        source_file_path = os.path.basename(self.source_file_path)
        filename, ext = os.path.splitext(source_file_path)

        self.audio_file_no_ext = filename.replace(' ', '_')

        self.temp_result_folder = os.path.join(TEMP_RESULT_FOLDER, self.audio_file_no_ext)

        audio_filename = '{}.{}'.format(self.audio_file_no_ext, 'wav')

        self.audio_file_path = os.path.join(AUDIO_FOLDER, audio_filename)

        self.config.diarizer.out_dir = self.temp_result_folder

    async def set_task_data(self, task_id, **kwargs):
        if self.status_writer:
            await self.status_writer.set_task_data(task_id, **kwargs)

    def _set_asr_model(self):
        if not os.path.exists(self.asr_model_path):
            asr_model = EncDecCTCModel.from_config_file(os.path.join(MODEL_DATA_FOLDER, 'ctc_model_config.yaml'))
            ckpt = torch.load(os.path.join(MODEL_DATA_FOLDER, 'ctc_model_weights.ckpt'), map_location="cpu")
            asr_model.load_state_dict(ckpt, strict=False)

            asr_model.save_to(self.asr_model_path)

    def _get_transcibing_model(self):
        model = EncDecRNNTBPEModel.from_config_file(os.path.join(MODEL_DATA_FOLDER, "rnnt_model_config.yaml"))
        ckpt = torch.load(os.path.join(MODEL_DATA_FOLDER, "rnnt_model_weights.ckpt"), map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model = model.to(DEVICE)
        return model

    async def process(self, task_id, file_path):

        # Preparing
        self._set_source_file_path(file_path)
        if self.is_video:
            logger.info('Converting audio file to audio mono 16 kGz. Start') 
         
            await self._convert_video_to_audio()
            await self.set_task_data(task_id, progress=5, upload_progress=100)  
            logger.info('Converting audio file to audio mono 16 kGz. Done')        
        else: 
            logger.info('Copying file to audio folder. Start') 
            self._copy_source_to_audio_folder()
            logger.info('Copying file to audio folder. Done') 

                    
        
        torch.cuda.empty_cache()

        if self.diarize:
            # ASR inference for words and word timestamps
            logger.info('Transcribing. Start') 
            asr_decoder_ts = GigaASRDecoderTimeStamps(self.config.diarizer, self.audio_file_path, status_writer=self.status_writer)
            asr_model = asr_decoder_ts.set_asr_model()
            word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
            await self.set_task_data(task_id, progress=70, transcribe_progress=100)  
            logger.info('Transcribing. Done')        

            logger.info('Diarizing. Start') 
            # Create a class instance for matching ASR and diarization results
            asr_diar_offline = GigaDiarWithASR(self.config.diarizer)
            asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset

            # Diarization inference for speaker labels
            diar_hyp, diar_score = asr_diar_offline.run_diarization(self.config, word_ts_hyp)
            trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
            await self.set_task_data(task_id, progress=100, diarization_progress=100)          
            logger.info('Diarizing. Done')    

            result = trans_info_dict[self.audio_file_no_ext]['sentences']     
        else:
            logger.info('Transcribing. Start')
            model = self._get_transcibing_model()

            # Initialize pyannote pipeline
            pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", 
            # use_auth_token=HF_TOKEN, 
            cache_dir=MODEL_DATA_FOLDER)
            pipeline = pipeline.to(DEVICE)

            batch_size = 2
            segments, boundaries = segment_audio(self.audio_file_path, pipeline)
            transcriptions = model.transcribe(segments, batch_size=batch_size)

            result = (transcriptions, boundaries)

        return result
    

def audiosegment_to_numpy(audiosegment: AudioSegment) -> np.ndarray:
    """Convert AudioSegment to numpy array."""
    samples = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32, order="C") / 32768.0
    return samples  

def segment_audio(
                    audio_path: str,
                    pipeline: Pipeline,
                    max_duration: float = 22.0,
                    min_duration: float = 15.0,
                    new_chunk_threshold: float = 0.2,
                ) -> Tuple[List[np.ndarray], List[List[float]]]:
    # Prepare audio for pyannote vad pipeline
    audio = AudioSegment.from_wav(audio_path)
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    # Process audio with pipeline to obtain segments with speech activity
    sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

    segments = []
    curr_duration = 0
    curr_start = 0
    curr_end = 0
    boundaries = []

    # Concat segments from pipeline into chunks for asr according to max/min duration
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(audio) / 1000, segment.end)
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):
            audio_segment = audiosegment_to_numpy(
                audio[curr_start * 1000 : curr_end * 1000]
            )
            segments.append(audio_segment)
            boundaries.append([curr_start, curr_end])
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        audio_segment = audiosegment_to_numpy(
            audio[curr_start * 1000 : curr_end * 1000]
        )
        segments.append(audio_segment)
        boundaries.append([curr_start, curr_end])

    return segments, boundaries


    