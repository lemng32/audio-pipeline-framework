import torch

from pyannote.audio import Pipeline
from tqdm import tqdm
from typing import Optional
from pathlib import Path
from vivoice_preprocess.preprocess_audio import (
  merge_audio,
  save_processed_audio,
  filter_by_diariazation,
)
from vivoice_preprocess.dataset_utils import (
  load_vivoice,
  load_vivoice_from_disk,
  save_to_dataset,
  filter_by_channel,
)

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
      out_audio_path (str, optional): The path the audios will be download to. Defaults to None.
      out_dataset_path (Optional[str], optional): The output directory for the created dataset. Defaults to None.
      token (str, optional): Huggingface token. Defaults to None.
      load_from_disk (Optional[bool], optional): Whether the datset is loaded from local files. Defaults to False.
      dataset_disk_path (Optional[str], optional): The path of the saved 'capleaf/viVoice' dataset, .arrow format. Defaults to None.
      load_cache_dir (Optional[str], optional): Path for dataset's cache directory. Defaults to None, which will use ~/.cache/huggingface.
  """
  if load_from_disk and not Path(dataset_disk_path).expanduser().is_dir():
    raise FileNotFoundError(f"Dataset input path: {dataset_disk_path} not found")

  if not Path(out_audio_path).expanduser().resolve().is_dir():
    Path(out_audio_path).mkdir(parents=True, exist_ok=True)

  if save_dataset_to_disk and not Path(out_dataset_path).expanduser().resolve().is_dir():
    Path(out_dataset_path).mkdir(parents=True, exist_ok=True)

  create_diarization_pipe(token=token)

  if load_from_disk:
    ds = load_vivoice_from_disk(ds_path=dataset_disk_path)
  else:
    ds = load_vivoice(token=token, cache_dir=load_cache_dir)

  channels = ds.unique("channel")
  # Hard-coded channels
  tmp = [
    channels[channels.index("@tamhonanuong")],
    channels[channels.index("@Nhantaidaiviet")],
  ]
  for channel in tqdm(tmp):
    cur_channel_ds = filter_by_channel(ds=ds, channel=channel)
    audio_df = merge_audio(ds=cur_channel_ds)
    if diarize_filter:
      audio_df = filter_by_diariazation(df=audio_df, dia_pipe=DIA_PIPE)
    save_processed_audio(channel=channel, df=audio_df, out_audio_path=out_audio_path)
    if save_dataset_to_disk:
      save_to_dataset(df=audio_df, out_dataset_path=out_dataset_path, channel=channel)
