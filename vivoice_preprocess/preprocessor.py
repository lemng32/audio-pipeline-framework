import pandas as pd
import numpy as np
import torch

from typing import Optional
from pydub import AudioSegment
from pathlib import Path
from utils.logger import get_logger
from vivoice_preprocess.whipser_asr import FasterWhisperASR


audio_count = 0


class PreprocessorPipeline:
  def __init__(self, dia_pipe, asr_model: FasterWhisperASR):
    self.logger = get_logger(__name__)
    self.pipe = dia_pipe
    self.model = asr_model

  def __call__(self, audio, sample_rate):
    # Step 1: Standardization
    audio = self.standardization(audio=audio, sample_rate=sample_rate)

    # Step 2: Diarization
    diarize_df = self.diarization(audio)

    # Step 3: ASR
    segments, info = self.whisper_asr(audio["waveform"])

    # TO-DO: Modify return type for general use. Currently return everything as a dict.
    return {
      "audio": audio,
      "diarize_df": diarize_df,
      "segments": segments,
      "info": info,
    }

  # Step 1: Standardization
  def standardization(self, audio, sample_rate: int):
    global audio_count

    if isinstance(audio, np.ndarray):
      waveform = audio
      name = f"audio_{audio_count:05}"
      audio_count += 1
    else:
      if isinstance(audio, Path):
        name = audio.name
        audio = AudioSegment.from_file(file=audio, format="wav")
      elif isinstance(audio, AudioSegment):
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
    waveform /= max_amplitude

    return {"waveform": waveform, "name": name, "sample_rate": sample_rate}

  # Step 2: Speaker diarization
  def diarization(self, audio):
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
  def whisper_asr(self, audio, options: Optional[dict] = None):
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

    return self.model.transcribe_audio(audio=audio, **transcription_options)
