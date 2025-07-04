from typing import Iterator
import pandas as pd

from datasets import load_dataset, Dataset, Audio
from pathlib import Path
from utils.file_utils import resolve_path, change_working_dir


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


def filter_by_channel(ds: Dataset, channel: str) -> Dataset:
  """
  Filters a dataset to include only rows with a specific channel value.

  Args:
    ds (Dataset): The Hugging Face dataset to filter.
    channel (str): The target channel name.

  Returns:
    Dataset: A filtered dataset containing only rows with the given channel.
  """
  return ds.filter(lambda batch: [c == channel for c in batch["channel"]], batched=True)


def create_and_save_dataset(
  df: pd.DataFrame,
  channel: str,
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
  out_dataset_path = resolve_path(out_dataset_path / channel)

  with (change_working_dir(saved_audio_path)):
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.rename_column(original_column_name="file_name", new_column_name="audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    ds.save_to_disk(dataset_path=out_dataset_path, max_shard_size=shard_size)


def slice_dataset(ds: Dataset, sample_rate: int = 24000, chunk_length: int = 10) -> Iterator:
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

  for j, row in enumerate(ds):
    length_window_sum += len(row["audio"]["array"])
    if length_window_sum >= sample_rate * chunk_length:
      yield ds[i:j+1]
      i = j + 1
      length_window_sum = 0