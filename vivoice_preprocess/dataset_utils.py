import pandas as pd
import os

from typing import Optional
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset, Audio


def load_vivoice(token: str, cache_dir: Optional[str] = None) -> Dataset:
  """
  Load capleaf/viVoice using huggingface's Dataset through the Hub.

  Args:
      token (str): Huggingface token
      cache_dir (Optional[str], optional): Path for dataset's cache directory. Defaults to None, which will use ~/.cache/huggingface.

  Returns:
      Dataset: The loaded dataset.
  """
  ds = load_dataset(
    "capleaf/viVoice",
    split="train",
    token=token,
    cache_dir=cache_dir,
  )
  return ds


def load_vivoice_from_disk(ds_path: str) -> Dataset:
  """
  Load capleaf/viVoice using huggingface's Dataset from_file(). Make sure that the dataset is saved under .arrow format.

  Args:
      ds_path (str): The path of the saved 'capleaf/viVoice' dataset, .arrow format.

  Returns:
      Dataset: The loaded dataset.
  """
  if not Path(ds_path).is_dir():
    print("cac")
    return None
  files = [
    f for f in Path(ds_path).iterdir()
    if f.is_file() and f.name.startswith("parquet") and f.name.endswith("arrow")
  ]
  ds = concatenate_datasets([Dataset.from_file(f"{file}") for file in files])
  return ds


def filter_by_channel(ds: Dataset, channel: str) -> Dataset:
  """_summary_

  Args:
      ds (Dataset): _description_
      channel (str): _description_

  Returns:
      Dataset: _description_
  """
  return ds.filter(
    lambda batch: [c == channel for c in batch["channel"]], batched=True
  ).sort(column_names="text")


def load_by_channel():
  return -1


def save_to_dataset(
  df: pd.DataFrame,
  out_dataset_path: str,
  out_audio_path: str,
  channel: str,
):
  """
  Create a dataset from the provided DataFrame and save it to disk.

  Args:
      df (pd.DataFrame): The provided pandas DataFrame.
      save_to_disk (Optional[bool]): Whether to save the dataset to disk.
      out_dataset_path (Optional[str]): The output directory for the created dataset.
      channel (Optional[str]): The channel attributed to the DataFrame.

  Returns:
      Dataset: The created dataset.
  """
  os.chdir(f"{out_audio_path}/{channel}")
  ds = Dataset.from_pandas(df[["channel", "text", "audio"]], preserve_index=False).cast_column("audio", Audio())
  ds.save_to_disk(dataset_path=f"{out_dataset_path}/{channel}", max_shard_size="500MB")
