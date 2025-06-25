import pandas as pd
import numpy as np
import torch
import soundfile as sf

from datasets import Dataset
from pyannote import Pipeline
from tqdm import tqdm
from typing import Optional
from pathlib import Path

SAMPLE_RATE = 24000
CHUNK_LENGTH = 10
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE


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
    array = np.pad(array, (0, int(SAMPLE_RATE * 0.5))).astype(np.float32)

    arrays_to_concat.append(array)
    texts_to_concat.append(row["text"])
    shape_sum += array.shape[0]

    if shape_sum >= N_SAMPLES:
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

  if arrays_to_concat and shape_sum >= N_SAMPLES:
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


def save_processed_audio(
  channel: str, df: pd.DataFrame, out_audio_path: str, drop_audio: Optional[bool] = True
):
  """
  Save the merged audios to disk.

  Args:
      channel (str): The channel name that is attributed to the audio.
      df (pd.DataFrame): The DataFrame that contains the merged audios.
      out_audio_path (str): The path the audios will be download to.
  """

  path = Path(f"{out_audio_path}/{channel}")
  if not path.is_dir():
    path.mkdir(parents=True, exist_ok=True)

  for audio in tqdm(df["audio"]):
    out_file = Path(path / f"{audio['path']}")
    if not out_file.isfile():
      sf.write(out_file, audio["array"], samplerate=SAMPLE_RATE)

  if drop_audio:
    df["audio"] = [audio["path"] for audio in df["audio"]]
    df = df.rename(columns={"audio": "file_name"})
    df[["file_name", "channel", "text"]].to_csv("metadata.csv", index=False)
  else:
    df["file_name"] = [audio["path"] for audio in df["audio"]]
    df[["file_name", "channel", "text"]].to_csv("metadata.csv", index=False)


def filter_by_diariazation(df: pd.DataFrame, dia_pipe: Pipeline) -> pd.DataFrame:
  """
  Remove any audios that have more than 1 potential speaker.

  Args:
      df (pd.DataFrame): _description_
      dia_pipe (Pipeline): _description_

  Returns:
      pd.DataFrame: _description_
  """
  idxs = []

  for i in tqdm(range(len(df))):
    cur_audio = df.iloc[i]["audio"]
    cac = torch.tensor(cur_audio["array"]).unsqueeze(0)
    diarization = dia_pipe({"waveform": cac, "sample_rate": SAMPLE_RATE})
    speakers = set(
      speaker for _, _, speaker in diarization.itertracks(yield_label=True)
    )
    if len(speakers) > 1:
      idxs.append(i)

  filtered_df = df.drop(idxs)
  filtered_df.reset_index(drop=True, inplace=True)

  return filtered_df
