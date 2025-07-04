import torch
import pandas as pd

from pyannote.audio import Pipeline
from tqdm import tqdm
from typing import Optional
from utils.file_utils import resolve_path
from utils.dataset_utils import (
  load_vivoice,
  create_and_save_dataset,
  filter_by_channel,
  slice_dataset,
)
from utils.logger import get_logger
from vivoice_preprocess.audio_handler import VivoiceAudioHandler


class VivoicePreprocessor:
  """
  Preprocess the capleaf/viVoice dataset by slicing, optional diarization, and saving processed data.
  """
  def __init__(self, out_audio_path: str, token: str, use_diarization: bool = True):
    self.out_audio_path = resolve_path(out_audio_path)
    self.token = token
    self.pipe = self._create_diarization_pipe(token) if use_diarization else None

  def _create_diarization_pipe(self, token: str) -> Pipeline:
    """
    Initialize and return a pyannote speaker diarization pipeline.
    Args:
        token (str): Hugging Face auth token

    Returns:
        Pipeline: Pyannote diarization pipeline
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
    Run the processing pipeline:
    1. Load the dataset capleaf/viVoice
    2. Merge short audio segments into one longer one
    3. Diarize the merged audio and remove audios with more than 1 potential speaker
    4. Save the processed audio with a metadata file
    3. [Optional]: Create and save a huggingface dataset based on the metadata

    Args:
        save_to_dataset (Optional[bool], optional): Whether to save audio as a Hugging Face dataset. Defaults to False.
        out_dataset_path (Optional[str], optional): Where to store the saved dataset if enabled. Defaults to None.
    """
    if save_to_dataset:
      out_dataset_path = resolve_path(out_dataset_path)

    ds = load_vivoice(token=self.token)
    channels = ds.unique("channel")

    audio_handler = VivoiceAudioHandler()

    logger = get_logger(__name__)

    # Hard-coded channels for quick testing
    # tmp = [channels[channels.index("@truyenhinhlaocai")]]
    tmp = [channels[channels.index("@khalid_dinh")]]
    for channel in tqdm(tmp, desc="Processing channel: "):
      channel_out_path = resolve_path(self.out_audio_path / channel)
      metadata = pd.DataFrame(columns=["file_name", "channel", "text"])

      cur_channel_ds = filter_by_channel(ds=ds, channel=channel)
      splits = slice_dataset(ds=cur_channel_ds)

      for split in tqdm(splits, desc="Processing channel splits: "):
        file_name = audio_handler(
          audios=[audio["array"] for audio in split["audio"]],
          sample_rate=split["audio"][0]["sampling_rate"],
          channel=channel,
          out_path=channel_out_path,
          pyannote_pipe=self.pipe
        )
        if file_name:
          #TO-DO: RUn Whisper ASR
          # -- Code goes here --
          metadata.loc[len(metadata)] = {
            "file_name": file_name,
            "channel": channel,
            "text": split["text"], 
          }
      metadata.to_csv(path_or_buf=channel_out_path / "metadata.csv", index=False)

      if save_to_dataset:
        create_and_save_dataset(
          df=metadata,
          channel=channel,
          out_dataset_path=out_dataset_path,
          saved_audio_path=channel_out_path,
        )
      
      logger.info(f"Saved {len(metadata["file_name"])} audio files at {channel_out_path}")
      logger.info(f"Finished processing channel: {channel}")
