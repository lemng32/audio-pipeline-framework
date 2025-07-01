import torch

from pyannote.audio import Pipeline
from tqdm import tqdm
from typing import Optional
from utils.utils import resolve_path
from utils.audio_utils import (
  merge_audio,
  save_processed_audio,
  filter_by_diariazation,
)
from utils.dataset_utils import (
  load_vivoice,
  create_and_save_dataset,
  filter_by_channel,
)


class VivoicePreprocessor:
  def __init__(self, out_audio_path: str, token: str, use_diarization: bool = True):
    self.out_audio_path = resolve_path(out_audio_path)
    self.token = token
    self.pipe = self._create_diarization_pipe(token) if use_diarization else None

  def _create_diarization_pipe(self, token: str) -> Pipeline:
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

  def run(
    self,
    save_to_dataset: Optional[bool] = False,
    out_dataset_path: Optional[str] = None,
  ):
    """
    Main driver function. Preprocess the audio and save them to the corresponding channel folder.
    The process currently includes:
      - Merge audio segments upto the specified length.
      - [Optional]: Run Pyannote's Speaker Diarization to remove audios with potentially more than 1 speaker.
      - Save the processed audios to disk.
      - [Optional]: Create a dataset to store the audios with its metadata and save it to disk.

    Args:
        save_to_dataset (Optional[bool], optional): Whether to save the dataset to disk. Defaults to False.
        out_dataset_path (Optional[str], optional): The output path for the created dataset. Defaults to None.

    Raises:
        FileNotFoundError: Dataset input path not found
        ValueError: Dataset output path not defined
    """
    if save_to_dataset:
      out_dataset_path = resolve_path(out_dataset_path)

    ds = load_vivoice(token=self.token)

    channels = ds.unique("channel")
    # Hard-coded channels for quick testing
    # tmp = [channels[channels.index("@truyenhinhlaocai")]]
    tmp = [channels[channels.index("@LamDongTV1")]]
    for channel in tqdm(tmp, desc="Processing current channel: "):
      cur_channel_ds = filter_by_channel(ds=ds, channel=channel)
      audio_df = merge_audio(ds=cur_channel_ds)
      if self.pipe:
        audio_df = filter_by_diariazation(df=audio_df, dia_pipe=self.pipe)
      save_processed_audio(
        df=audio_df,
        channel=channel,
        out_audio_path=self.out_audio_path
      )
      if save_to_dataset:
        create_and_save_dataset(
          df=audio_df,
          channel=channel,
          out_dataset_path=out_dataset_path,
          saved_audio_path=self.out_audio_path,
        )
