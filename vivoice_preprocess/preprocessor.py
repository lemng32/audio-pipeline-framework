import pandas as pd
import numpy as np
import torch

from pydub import AudioSegment
from pathlib import Path
from pyannote.audio import Pipeline
# from tqdm import tqdm
# from typing import Iterator, Optional
# from utils.file_utils import resolve_path
# from utils.logger import get_logger
# from vivoice_preprocess.whipser_asr import FasterWhisperASR

class PreprocessorPipeline:

  def __init__(self, token):
    self.pipe = Pipeline.from_pretrained(
      "pyannote/speaker-diarization-3.1",
      use_auth_token=token,
    )
    self.pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    self.audio_count = 0
    #self.model = FasterWhisperASR(model_size="turbo")

  def __call__(self, audio, sample_rate):
    # Step 1: Standardization
    audio = self.standardization(audio=audio, sample_rate=sample_rate)
    # Step 2: Diarization
    diarize_df = self.diarization(audio)
    # TO-DO: ASR

    # TO-DO: Modify return type for general use. Currently coded to be used for vivoice.
    return {
      "audio": audio,
      "diarize_df": diarize_df
    }

  # Step 1: Standardization
  def standardization(self, audio, sample_rate: int):
 
    if isinstance(audio, np.ndarray):
      waveform = audio
      name = f"audio_{self.audio_count:05}"
      self.audio_count += 1
    else:
      if isinstance(audio, Path):
        name = audio.name
        audio = AudioSegment.from_file(file=audio, format="wav")
      elif isinstance(audio, AudioSegment):
        name = f"audio_{self.audio_count:05}"
        self.audio_count += 1
      else:
        print("cac")

      audio = audio.set_frame_rate(sample_rate)
      audio = audio.set_sample_width(2)
      audio = audio.set_channels(1)

      target_dBFS = -20
      gain = target_dBFS - audio.dBFS

      normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

      waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)

    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude

    return {
      "waveform": waveform,
      "name": name,
      "sample_rate": sample_rate
    }

  # Step 2: Speaker diarization
  def diarization(self, audio):
    annotation = self.pipe(
      {
        "waveform": torch.tensor(audio["waveform"]).unsqueeze(0),
        "sample_rate": audio["sample_rate"],
        "channel": 0
      }
    )
    diarize_df = pd.DataFrame(
      data=[(segment, track, label) for segment, track, label in annotation.itertracks(yield_label=True)],
      columns=["segment", "track", "speaker"]
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
    return diarize_df
  
  # Step 3: Whipser ASR
  def whisper_asr(self, audio):
    return self.model.transcribe_audio(audio["waveform"])