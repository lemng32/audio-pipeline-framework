import pandas as pd

from datasets import load_dataset, Dataset, Audio
from pathlib import Path
from utils.utils import change_working_dir


def load_vivoice(token: str) -> Dataset:
  """
  Load capleaf/viVoice using huggingface's Dataset through the Hub.

  Args:
      token (str): Huggingface token

  Returns:
      Dataset: The loaded dataset.
  """
  ds = load_dataset(
    "capleaf/viVoice",
    split="train",
    token=token,
  )
  return ds


def filter_by_channel(ds: Dataset, channel: str) -> Dataset:
  """
  Filters the dataset by the specified channel.

  Args:
      ds (Dataset): The loaded dataset.
      channel (str): The channel to filter by.

  Returns:
      Dataset: A dataset containing only entries of the filtered channel.
  """
  return ds.filter(lambda batch: [c == channel for c in batch["channel"]], batched=True)


# TO-DO: Implement loading processed audio folder into dataset using metadata
def load_by_channel():
  return -1


def create_and_save_dataset(
  df: pd.DataFrame,
  channel: str,
  out_dataset_path: Path,
  saved_audio_path: Path,
):
  """
  Create a dataset from the provided dataframe and save it to disk.

  Note: Assumes that audio files referenced in the dataframe are already saved to disk.

  Args:
      df (pd.DataFrame): The provided dataframe.
      channel (str): The channel name that is attributed to the audio
      out_dataset_path (Path): The output path for the created dataset.
      saved_audio_path (Path): The path oif the saved audios.
  """
  out_dataset_path = out_dataset_path / channel
  saved_audio_path = saved_audio_path / channel

  with (change_working_dir(saved_audio_path)):
    ds = Dataset.from_pandas(
      df[["channel", "text", "audio"]], preserve_index=False
    ).cast_column("audio", Audio())
    ds.save_to_disk(dataset_path=out_dataset_path, max_shard_size="500MB")
