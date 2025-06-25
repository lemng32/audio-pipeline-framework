import torch

from pyannote.audio import Pipeline
from tqdm import tqdm
from typing import Optional
from pathlib import Path
from vivoice_preprocess.audio_utils import (
  merge_audio,
  save_processed_audio,
  filter_by_diariazation,
)
from vivoice_preprocess.dataset_utils import (
  load_vivoice,
  load_vivoice_from_disk,
  create_and_save_dataset,
  filter_by_channel,
)


def create_diarization_pipe(token: str = None) -> Pipeline:
  """
  Instantiate Pyannote's Diarization Pipeline.

  Args:
      token (str, optional): Huggingface token. Defaults to None.

  Returns:
      Pipeline: The diarization pipeline instance.
  """
  pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
  )
  pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
  return pipe


def preprocess(
  out_audio_path: str,
  token: str = None,
  load_from_disk: Optional[bool] = False,
  dataset_disk_path: Optional[str] = None,
  load_cache_dir: Optional[str] = None,
  save_dataset_to_disk: Optional[bool] = False,
  out_dataset_path: Optional[str] = None,
  diarize_filter: Optional[bool] = True,
):
  """
  Main driver function. Preprocess the audio and save them to the corresponding channel folder.

  Args:
      out_audio_path (str): The output path for the processed audio
      token (str, optional): Huggingface token. Defaults to None.
      load_from_disk (Optional[bool], optional): Whether to load the dataset from local files. Defaults to False.
      dataset_disk_path (Optional[str], optional): The path where the dataset files are stored, in .arrow format. Defaults to None.
      cache_dir (Optional[str], optional): Path for dataset's cache directory. Defaults to None, which will use ~/.cache/huggingface.
      save_dataset_to_disk (Optional[bool], optional): Whether to save the dataset to disk. Defaults to False.
      out_dataset_path (Optional[str], optional): The output path for the created dataset. Defaults to None.
      diarize_filter (Optional[bool], optional): Whether to run diarization to filter the audios. Defaults to True.

  Raises:
      FileNotFoundError: Dataset input path not found
      ValueError: Dataset output path not defined
  """

  if load_from_disk and not dataset_disk_path:
    raise FileNotFoundError(f"Dataset input path: {dataset_disk_path} not found")

  out_audio_path = str(Path(out_audio_path).expanduser().resolve())
  if not Path(out_audio_path).is_dir():
    Path(out_audio_path).mkdir(parents=True, exist_ok=True)

  if save_dataset_to_disk:
    if not out_dataset_path:
      raise ValueError("Dataset output path not defined")
    else:
      out_dataset_path = str(Path(out_dataset_path).expanduser().resolve())
      if not Path(out_dataset_path).is_dir():
        Path(out_dataset_path).mkdir(parents=True, exist_ok=True)

  pipe = create_diarization_pipe(token=token)

  if load_from_disk:
    ds = load_vivoice_from_disk(ds_path=dataset_disk_path)
  else:
    ds = load_vivoice(token=token, cache_dir=load_cache_dir)

  channels = ds.unique("channel")
  # Hard-coded channels
  # tmp = [
  #   channels[channels.index("@tamhonanuong")],
  #   channels[channels.index("@Nhantaidaiviet")],
  # ]
  tmp = [channels[channels.index("@PhimHOTTK-L")]]
  for channel in tqdm(tmp, desc="Processing current chanel: "):
    cur_channel_ds = filter_by_channel(ds=ds, channel=channel)
    audio_df = merge_audio(ds=cur_channel_ds)
    if diarize_filter:
      audio_df = filter_by_diariazation(df=audio_df, dia_pipe=pipe)
    save_processed_audio(channel=channel, df=audio_df, out_audio_path=out_audio_path)
    if save_dataset_to_disk:
      create_and_save_dataset(
        df=audio_df,
        out_dataset_path=out_dataset_path,
        out_audio_path=out_audio_path,
        channel=channel
      )
