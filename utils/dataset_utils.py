import pandas as pd
import os

from datasets import (
  load_dataset,
  Dataset,
  Audio
)


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
  return ds.filter(
    lambda batch: [c == channel for c in batch["channel"]], batched=True
  ).sort(column_names="text")


# TO-DO: Implement loading processed audio folder into dataset using metadata
def load_by_channel():
  return -1


def create_and_save_dataset(
  df: pd.DataFrame,
  channel: str,
  out_dataset_path: str,
  out_audio_path: str,
):
  """
  Create a dataset from the provided dataframe and save it to disk.

  Args:
      df (pd.DataFrame): The provided dataframe.
      channel (str): The channel name that is attributed to the audio
      out_dataset_path (str): The output path for the created dataset.
      out_audio_path (str): The output path for the processed audio.
  """
  os.chdir(f"{out_audio_path}/{channel}")
  ds = Dataset.from_pandas(
    df[["channel", "text", "audio"]], preserve_index=False
  ).cast_column("audio", Audio())
  ds.save_to_disk(dataset_path=f"{out_dataset_path}/{channel}", max_shard_size="500MB")
