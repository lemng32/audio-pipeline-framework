import os
import pandas as pd
import numpy as np
import torch
import soundfile as sf

from datasets import (
  Dataset,
  Audio,
)
from pyannote.audio import Pipeline
from tqdm import tqdm
from typing import Optional
from load import load_vivoice, load_vivoice_from_disk

SAMPLE_RATE = 24000
CHUNK_LENGTH = 10
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
DIA_PIPE = None


def create_diarization_pipe(token: str = None) -> bool:
  """
  Initialize Pyannote's Diarization Pipeline.

  Args:
      token (str, optional): Huggingface token. Defaults to None.

  Returns:
      bool: True if the pipeline is successfully initialized.
  """
  DIA_PIPE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
  )
  DIA_PIPE.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
  return True

def filter_by_channel(batch: dict, channel: str) -> list[bool]:
  """
  Filters the dataset by the specified channel.

  Args:
      batch (dict): The current batch.
      channel (str): The channel to be filtered by.

  Returns:
      list[bool]: A list off booleans.
  """
  return [c == channel for c in batch["channel"]]


def merge_audio(ds: Dataset) -> pd.DataFrame:
  """
  Merge short audios (given in numpy.array) into one longer audio.

  Args:
      ds (Dataset): The current dataset.

  Returns:
      pd.DataFrame: A pandas Dataframe of the same structure, with concatenated audios and transcipts.
  """
  arrays_to_concat = []
  texts_to_concat = []
  shape_sum = 0
  df = pd.DataFrame(columns=["channel", "text", "audio"])

  for row in tqdm(ds):
    array = row["audio"]["array"]
    array = np.pad(array, (0, int(SAMPLE_RATE * 0.5)))

    arrays_to_concat.append(array)
    texts_to_concat.append(row["text"])
    shape_sum += array.shape[0]

    if shape_sum > N_SAMPLES:
      df.loc[len(df)] = {
        "channel": row["channel"],
        "text": " ".join(texts_to_concat),
        "audio": {
          "path": row["audio"]["path"],
          "array": np.array(np.concatenate(arrays_to_concat)),
          "sampling_rate": row["audio"]["sampling_rate"],
        },
      }
      shape_sum = 0
      arrays_to_concat.clear()
      texts_to_concat.clear()

  if arrays_to_concat:
    df.loc[len(df)] = {
      "channel": row["channel"],
      "text": " ".join(texts_to_concat),
      "audio": {
        "path": row["audio"]["path"],
        "array": np.array(np.concatenate(arrays_to_concat)),
        "sampling_rate": row["audio"]["sampling_rate"],
      },
    }

  return df


def save_audio(channel: str, df: pd.DataFrame, out_audio_path: str):
  """
  Save the merged audios to disk.

  Args:
      channel (str): The channel name that is attributed to the audio.
      df (pd.DataFrame): The dataframe that contains the merged audios.
      out_audio_path (str): The path the audios will be download to
  """
  if not os.path.isdir(f"{out_audio_path}/{channel}"):
    os.makedirs(f"{out_audio_path}/{channel}")

  for audio in tqdm(df["audio"]):
    out_file = f"{out_audio_path}/{channel}/{audio['path']}"
    if not os.path.isfile(out_file):
      sf.write(out_file, audio["array"], samplerate=SAMPLE_RATE)


def filter_by_diariazation(df: pd.DataFrame) -> pd.DataFrame:
  """
  Remove any audios that have more than 1 potential speaker.

  Args:
      df (pd.DataFrame): _description_

  Returns:
      pd.DataFrame: _description_
  """
  idxs = []

  for i in tqdm(range(len(df))):
    cur_audio = df.iloc[i]["audio"]
    cac = torch.tensor(cur_audio["array"].astype(np.float32)).unsqueeze(0)
    diarization = DIA_PIPE({"waveform": cac, "sample_rate": SAMPLE_RATE})
    speakers = set(
      speaker for _, _, speaker in diarization.itertracks(yield_label=True)
    )
    if len(speakers) > 1:
      idxs.append(i)

  filtered_df = df.drop(idxs)
  filtered_df.reset_index(drop=True, inplace=True)

  return filtered_df


def create_dataset(
  df: pd.DataFrame,
  save_to_disk: Optional[bool],
  out_dataset_path: Optional[str],
  channel: Optional[str],
) -> Dataset:
  """_summary_

  Args:
      df (pd.DataFrame): _description_
      save_to_disk (Optional[bool]): _description_
      out_dataset_path (Optional[str]): _description_
      channel (Optional[str]): _description_

  Returns:
      Dataset: _description_
  """
  tmp_df = df.copy()
  tmp_df["audio"] = [audio["path"] for audio in tmp_df["audio"]]
  ds = Dataset.from_pandas(tmp_df, preserve_index=False).cast_column("audio", Audio())
  if save_to_disk:
    ds.save_to_disk(
      dataset_path=f"{out_dataset_path}/{channel}", max_shard_size="500MB"
    )
  return ds


def preprocess(
  out_audio_path: str = None,
  token: str = None,
  load_from_disk: Optional[bool] = False,
  dataset_disk_path: Optional[str] = None,
  load_cache_dir: Optional[str] = None,
  out_dataset_path: Optional[str] = None,
  save_to_disk: Optional[bool] = False,
):
  """_summary_

  Args:
      out_audio_path (str, optional): _description_. Defaults to None.
      token (str, optional): _description_. Defaults to None.
      load_from_disk (Optional[bool], optional): _description_. Defaults to False.
      dataset_disk_path (Optional[str], optional): _description_. Defaults to None.
      load_cache_dir (Optional[str], optional): _description_. Defaults to None.
      out_dataset_path (Optional[str], optional): _description_. Defaults to None.
      save_to_disk (Optional[bool], optional): _description_. Defaults to False.

  Returns:
      _type_: _description_
  """
  if load_from_disk and not os.path.isdir(dataset_disk_path):
    raise FileNotFoundError(f"Dataset input path: {dataset_disk_path} not found")
  if not out_audio_path:
    raise FileNotFoundError(f"Audio output path: {out_audio_path} not found.")
  if save_to_disk and not out_dataset_path:
    raise FileNotFoundError(f"Dataset output path: {dataset_disk_path} not found")
  
  create_diarization_pipe(token=token)

  if load_from_disk:
    ds = load_vivoice_from_disk(ds_path=dataset_disk_path)
  else:
    ds = load_vivoice(token=token, cache_dir=load_cache_dir)

  channels = ds.unique("channel")
  for channel in channels[:1]:
    cur_channel_ds = ds.filter(
      lambda batch: filter_by_channel(batch, channel), batched=True
    ).sort(column_names="text")
    audio_df = merge_audio(cur_channel_ds)
    # audio_df = filter_by_diariazation(audio_df)
    save_audio(channel=channel, df=audio_df, out_audio_path=out_audio_path)
    # create_dataset(df=audio_df, save_to_disk=save_to_disk, out_dataset_path=out_dataset_path, channel=channel)

  return True
