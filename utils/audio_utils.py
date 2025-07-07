import numpy as np
import soundfile as sf


def merge_audio(
  audios: list[np.ndarray],
  sample_rate: int = 24000,
  pad_width: float = 0.5
) -> np.ndarray:
  """
  Merge multiple audio arrays into one, padding each with silence.

  Args:
    audios (list[np.ndarray]): List of 1D audio arrays.
    sample_rate (int): Sample rate of the audio. Defaults to 24000.
    pad_width (float): Seconds of silence to pad between audio chunks. Defaults to 0.5 seconds.

  Returns:
    np.ndarray: Concatenated audio array.
  """
  merged_audios = []
  for audio in audios:
    merged_audios.append(
      np.pad(
        array=audio, pad_width=(0, int(sample_rate * pad_width))
      ).astype(np.float32)
    )
  return np.concatenate(merged_audios)


def save_audio(out_file: str, audio: np.ndarray, sample_rate: int = 24000) -> None:
  """
  Save a NumPy audio array to disk as a WAV file.

  Args:
    out_file (Path): Full path to save the file.
    audio (np.ndarray): 1D audio array.
    sample_rate (int): Sample rate of the audio. Defaults to 24000.
  """
  sf.write(file=out_file, data=audio, samplerate=sample_rate)
