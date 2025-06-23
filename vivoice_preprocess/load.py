import os

from typing import Optional
from datasets import (
  load_dataset,
  concatenate_datasets,
  Dataset,
)

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
  Load capleaf/viVoice using huggingface's Dataset from local .arrow files.

  Args:
      ds_path (str): Path for the .arrow files.

  Returns:
      Dataset: The loaded dataset.
  """
  if not os.path.isdir(ds_path):
    print("cac")
    return None
  files = [
    f for f in os.listdir(ds_path) if f.startswith("parquet") and f.endswith("arrow")
  ]
  ds = concatenate_datasets([Dataset.from_file(f"{ds_path}/{file}") for file in files])
  return ds