import numpy as np
import uuid
import torch

from pyannote.audio import Pipeline
from pathlib import Path
from typing import Optional
from utils.audio_utils import merge_audio, save_audio


class VivoiceAudioHandler:
  """
  Handles merging of audio segments and speaker diarization filtering for capleaf/viVoice.
  """
  def __call__(
    self,
    audios: list[np.ndarray],
    channel: str,
    out_path: Path,
    pyannote_pipe: Pipeline,
    sample_rate: int = 24000,
  ) -> Optional[str]:
    """
    Merge audio segments and optionally filter by number of speakers.

    Args:
      audios (list[dict]): List of audio arrays.
      channel (str): Current channel associated with the audios
      out_path (Path): Path to directory where output will be saved.
      pyannote_pipe (Pipeline): Pyannote pipeline for diarization.
      sample_rate (int): Sample rate of the audios. Defaults to 24000.

    Returns:
      Optional[str]: File name of saved audio, or None if filtered out.
    """
    file_name = f"{channel}_{uuid.uuid4().hex}.wav"

    merged_audio = merge_audio(audios=audios, sample_rate=sample_rate)
    if pyannote_pipe:
      num_speakers = self._get_num_speakers(audio=merged_audio, dia_pipe=pyannote_pipe)
      if num_speakers > 1:
        return None
    save_audio(audio=merged_audio, out_file=out_path / file_name)
    return file_name
  
  def _get_num_speakers(
    self, audio: np.ndarray, dia_pipe: Pipeline, sample_rate: int = 24000
  ) -> int:
    """
    Estimate the number of unique speakers in an audio segment using pyannote.

    Args:
      audio (np.ndarray): 1D audio array.
      dia_pipe (Pipeline): Preloaded pyannote pipeline.
      sample_rate (int): Sample rate of the audio. Defaults to 24000.

    Returns:
      int: Number of unique speakers detected.
    """
    waveform = torch.tensor(data=audio).unsqueeze(0)
    diarization = dia_pipe({"waveform": waveform, "sample_rate": sample_rate})
    speakers = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
    return len(speakers)