import pandas as pd
import numpy as np
import torch
import soundfile as sf
import uuid

from pyannote.audio import Pipeline
from pathlib import Path


class VivoiceAudioHandler:
  def __call__(
    self,
    audios: list[dict],
    out_path: Path,
    pyannote_pipe: Pipeline
  ) -> str:
    file_name = f"audio_{uuid.uuid4().hex}.wav"

    merged_audio = merge_audio([audio["array"] for audio in audios])
    if pyannote_pipe:
      num_speakers = get_num_speakers(audio=merged_audio, dia_pipe=pyannote_pipe)
      if num_speakers > 1:
        return None
    save_audio(
      audio=merged_audio,
      out_file=out_path / file_name
    )
    return file_name


def merge_audio(audios: list[np.array], sample_rate: int = 24000, pad_width: float = 0.5) -> np.array:
  merged_audios = []
  for audio in audios:
    merged_audios.append(np.pad(array=audio, pad_width=(0, int(sample_rate * pad_width))).astype(np.float32))
  return np.concatenate(merged_audios)


def save_audio(out_file: str, audio: np.array, sample_rate: int = 24000) -> None:
  sf.write(file=out_file, data=audio, samplerate=sample_rate)


def get_num_speakers(
  audio: np.array, dia_pipe: Pipeline, sample_rate: int = 24000
) -> pd.DataFrame:
  """
  Remove any audios that have more than 1 potential speaker.

  Args:
      df (pd.DataFrame): The provided dataframe.
      dia_pipe (Pipeline): The diarazation pipeline instance.
      sample_rate (int): The sample rate of the audio

  Returns:
      pd.DataFrame: The filtered dataframe.
  """
  waveform = torch.tensor(audio).unsqueeze(0)
  diarization = dia_pipe({"waveform": waveform, "sample_rate": sample_rate})
  speakers = set(
    speaker for _, _, speaker in diarization.itertracks(yield_label=True)
  )
  return len(speakers)