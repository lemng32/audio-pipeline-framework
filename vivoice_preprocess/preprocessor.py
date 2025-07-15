import pandas as pd
import numpy as np
import torch
import librosa

from typing import Optional, Union
from pydub import AudioSegment
from pathlib import Path
from pyannote.audio import Pipeline
from utils.logger import get_logger
from vivoice_preprocess.whisper_asr import FasterWhisperASR


audio_count = 0


class PreprocessorPipeline:
  """
  General pipeline to process audio
  """

  def __init__(self, dia_pipe: Pipeline, asr_model: FasterWhisperASR) -> None:
    """
    Initialize the PreprocessorPipeline.

    Args:
      dia_pipe (Pipeline): Preloaded pyannote.audio Pipeline for speaker diarization.
      asr_model (FasterWhisperASR): Custom FasterWhisper ASR wrapper.
    """
    self.logger = get_logger(__name__)
    self.pipe = dia_pipe
    self.model = asr_model

  def __call__(
    self, audio: Union[np.ndarray, str, AudioSegment], sample_rate: int = None
  ) -> dict:
    """
    Run the audio preprocessing pipeline.

    Args:
      audio (Union[np.ndarray, Path]): Audio waveform or path to audio file.
      sample_rate (int): Audio sample rate.

    Raises:
      ValueError: Sample rate of audio not provided

    Returns:
      dict: A dictionary:
      {
        "audio": Dictionary with waveform, name, and sample rate
        "diarize_df": Pandas DataFrame with speaker diarization segments
        "segments": List of ASR transcribed segments
        "info": Metadata returned from the ASR model
      }
    """
    if isinstance(audio, str):
      audio = Path(audio)
      if not audio.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio}")
    elif isinstance(audio, np.ndarray):
      if audio.ndim > 2:
        raise ValueError(f"Expected mono or stereo audio array, got shape: {audio.shape}")
    
    if not sample_rate:
      raise ValueError("Sample rate of audio not provided")
    
    # Step 1: Standardization
    audio = self.standardization(audio=audio, sample_rate=sample_rate)

    # Step 2: Diarization
    diarize_df = self.diarization(audio)

    # Step 3: ASR
    segments, info = self.whisper_asr(audio)

    # TO-DO: Modify return type for general use. Currently return everything as a dict.
    return {
      "audio": audio,
      "diarize_df": diarize_df,
      "segments": segments,
      "info": info,
    }

  # Step 1: Standardization
  def standardization(
    self, audio: Union[np.ndarray, Path, AudioSegment], sample_rate: int
  ) -> dict:
    """
    Standardize audio by resampling, normalizing volume, converting to mono, and extracting waveform.

    Args:
      audio (Union[np.ndarray, Path]): Input audio as a NumPy array or file path.
      sample_rate (int): Desired sample rate.

    Returns:
      dict: A dictionary:
      {
        "waveform": Numpy array (float32 normalized audio),
        "name": Audio file name or generated ID,
        "sample_rate": Sample rate,
      }
    """
    global audio_count

    if isinstance(audio, np.ndarray):
      waveform = audio
      name = f"audio_{audio_count:05}"
      audio_count += 1
    else:
      if isinstance(audio, Path):
        name = audio.name
        audio = AudioSegment.from_file(file=audio)
      elif isinstance(audio, (AudioSegment, np.ndarray)):
        name = f"audio_{audio_count:05}"
        audio_count += 1
      else:
        raise ValueError("Unsupported file type")

      audio = audio.set_frame_rate(sample_rate)
      audio = audio.set_sample_width(2)
      audio = audio.set_channels(1)

      target_dBFS = -20
      gain = target_dBFS - audio.dBFS

      normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

      waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)

    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
      waveform /= max_amplitude

    return {"waveform": waveform, "name": name, "sample_rate": sample_rate}

  # Step 2: Speaker diarization
  def diarization(self, audio: dict) -> pd.DataFrame:
    """
    Perform speaker diarization on audio.

    Args:
      audio (dict): Dictionary containing waveform and sample rate.

    Returns:
      pd.DataFrame: DataFrame with diarization info:
      {
        "segment": Time interval for a speech segment,
        "track": Track identifier within the segment,
        "speaker": Speaker label assigned by the diarization pipeline,
        "start": Start time in seconds,
        "end": End time in seconds,
      }.
    """
    annotation = self.pipe(
      {
        "waveform": torch.tensor(audio["waveform"]).unsqueeze(0),
        "sample_rate": audio["sample_rate"],
        "channel": 0,
      }
    )
    annotation_iter = annotation.itertracks(yield_label=True)

    diarize_df = pd.DataFrame(
      data=annotation_iter,
      columns=["segment", "track", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    return diarize_df

  # Step 3: Whipser ASR
  def whisper_asr(self, audio: dict, options: Optional[dict] = None) -> tuple:
    """
    Run Whisper ASR model on the given audio.

    Args:
      audio (np.ndarray): Audio object with waveform, name, and sample rate.
      options (dict, optional): Additional parameters for transcription.

    Returns:
      tuple: A tuple (segments, info) with transcription results and metadata.
    """
    transcription_options = {
      "beam_size": 5,
      "condition_on_previous_text": True,
      "without_timestamps": False,
      "vad_filter": True,
      "language": None,
      "batch_size": None,
    }

    if options:
      transcription_options.update(options)

    audio_waveform = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=16000)

    return self.model.transcribe_audio(audio=audio_waveform, **transcription_options)
