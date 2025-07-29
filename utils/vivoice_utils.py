import pandas as pd
import numpy as np
import re

from typing import Iterator
from datasets import load_dataset, Dataset, Audio
from pathlib import Path
from utils.file_utils import change_working_dir
from underthesea import text_normalize
from num2words import num2words
from jiwer import wer


def load_vivoice(token: str) -> Dataset:
  """
  Load the capleaf/viVoice dataset from the Hugging Face Hub.

  Args:
    token (str): Hugging Face token for authentication.

  Returns:
    Dataset: The loaded training split of the dataset.
  """
  ds = load_dataset(
    "capleaf/viVoice",
    split="train",
    token=token,
  )
  return ds


def filter_by_channel(dataset: Dataset) -> dict:
  """
  Filters a dataset to include only rows with a specific channel value.

  Args:
    ds (Dataset): The Hugging Face dataset to filter.
    channel (str): The target channel name.

  Returns:
    Dataset: A filtered dataset containing only rows with the given channel.
  """
  channels = dataset.unique(column="channel")
  index_dict = {channel: [] for channel in channels}
  for i, channel in enumerate(dataset["channel"]):
    index_dict[channel].append(i)
  return index_dict


def save_dataset(
  df: pd.DataFrame,
  out_dataset_path: Path,
  saved_audio_path: Path,
  sample_rate: int = 24000,
  shard_size: str = "500MB",
) -> None:
  """
  Create and save a Hugging Face Dataset from a DataFrame.

  Args:
    df (pd.DataFrame): A DataFrame with at least a 'file_name' column.
    channel (str): The channel label used to determine subdirectory.
    out_dataset_path (Path): Root directory to save the dataset.
    saved_audio_path (Path): Directory where audio files are stored.
    sample_rate (int): Expected sample rate of the audio files. Defaults to 24000
    shard_size (str): Max size per saved dataset shard. Defaults to 500MB
  """

  with change_working_dir(saved_audio_path):
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.rename_column(original_column_name="file_name", new_column_name="audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    ds.save_to_disk(dataset_path=out_dataset_path, max_shard_size=shard_size)


def slice_dataset(
  dataset: Dataset, sample_rate: int = 24000, chunk_length: int = 10
) -> Iterator:
  """
  Yields slices of the dataset, where each slice contains ~chunk_length seconds of audio.

  Args:
    ds (Dataset): Hugging Face dataset with audio column.
    sample_rate (int): Sample rate of the audio. Defaults to 24000
    chunk_length (int): Desired duration (in seconds) for each slice. Defaults to 10 seconds.

  Yields:
    Dataset: A slice of the original dataset.
  """
  length_window_sum = 0
  i = 0

  for j, row in enumerate(dataset):
    length_window_sum += len(row["audio"]["array"])
    if length_window_sum >= sample_rate * chunk_length:
      yield dataset[i : j + 1]
      i = j + 1
      length_window_sum = 0


def merge_audio(
  audios: list[np.ndarray], sample_rate: int = 24000, pad_width: float = 0.5
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
      np.pad(array=audio, pad_width=(0, int(sample_rate * pad_width))).astype(
        np.float32
      )
    )
  return np.concatenate(merged_audios)


def process_text(references: list, predictions: list) -> dict:
  """
  Process the given reference and prediction text.

  Args:
    references (list): The list of reference text.
    predictions (list): The list of ASR genererated text.

  Returns:
    dict: The dictionary to be added to the metadata, containing:
      {
        "text": The normalized reference,
        "gen_text": The normalized predictions,
        "wer": The WER score
      }
  """

  metadata_text = " ".join(references)
  metadata_gen_text = " ".join(predictions)

  norm_reference = normalize(metadata_text)
  norm_prediction = normalize(metadata_gen_text)

  score = wer(reference=norm_reference, hypothesis=norm_prediction)

  return {"text": norm_reference, "gen_text": norm_prediction, "wer": score}


def normalize(text) -> str:
  # Remove punctuation (keep numbers and letters)
  text = re.sub(r"[^\w\s]", "", text)

  # Normalize whitespace
  text = re.sub(r"\s+", " ", text).strip()

  # Convert number to text
  text = re.sub(r"\b\d+\b", lambda m: num2words(int(m.group()), lang="vi"), text)

  norm_text = text_normalize(text)
  norm_text = norm_text.lower()
  return norm_text
