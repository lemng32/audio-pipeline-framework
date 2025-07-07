import torch
import pandas as pd

from pathlib import Path
from pyannote.audio import Pipeline
from tqdm import tqdm
from typing import Iterator, Optional
from utils.file_utils import resolve_path
from utils.dataset_utils import (
  load_vivoice,
  save_dataset,
  filter_by_channel,
  slice_dataset,
)
from utils.logger import get_logger
from vivoice_preprocess.audio_handler import VivoiceAudioHandler


class VivoicePreprocessor:
  """
  Preprocess the capleaf/viVoice dataset by slicing, optional diarization, and saving processed data.
  """

  def __init__(
    self, out_audio_path: str, token: str, use_diarization: bool = True
  ) -> None:
    self.out_audio_path = resolve_path(out_audio_path)
    self.out_dataset_path = None
    self.token = token
    self.pipe = self._create_diarization_pipe(token) if use_diarization else None
    self.audio_handler = VivoiceAudioHandler()
    self.logger = get_logger(__name__)

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

  def _process_channel_split(
    self, splits: Iterator, channel: str, out_path: Path
  ) -> pd.DataFrame:
    """
    Process dataset splits for a specific channel.

    Args:
        splits (Iterator): Iterator of dataset splits.
        channel (str): Current channel name.
        out_path (Path): Output path for saving audio files.

    Returns:
        pd.DataFrame: Metadata DataFrame with file_name, channel, and text.
    """
    metadata = pd.DataFrame(columns=["file_name", "channel", "text"])

    for split in tqdm(splits, desc="Processing channel splits: "):
      file_name = self.audio_handler(
        audios=[audio["array"] for audio in split["audio"]],
        sample_rate=split["audio"][0]["sampling_rate"],
        channel=channel,
        out_path=out_path,
        pyannote_pipe=self.pipe,
      )
      if file_name:
        metadata.loc[len(metadata)] = {
          "file_name": file_name,
          "channel": channel,
          "text": split["text"],
        }
    return metadata

  def _save_metadata_to_dataset(
    self, skip: bool, metadata: pd.DataFrame, audio_path: Path
  ) -> None:
    """
    Optionally save the metadata to Hugging Face dataset format.

    Args:
        skip (bool): Whether to skip saving.
        metadata (pd.DataFrame): Metadata to save.
        audio_path (Path): Path where the audio files are stored.
    """
    if skip:
      self.logger.info("Skipping saving to dataset.")
    else:
      self.logger.info(f"Saving dataset to: {self.out_dataset_path}.")
      save_dataset(
        df=metadata,
        out_dataset_path=self.out_dataset_path,
        saved_audio_path=audio_path,
      )
      self.logger.info("Dataset saved successfully.")

  def run(
    self,
    save_to_dataset: Optional[bool] = False,
    out_dataset_path: Optional[str] = None,
  ) -> None:
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
      self.out_dataset_path = resolve_path(out_dataset_path)

    ds = load_vivoice(token=self.token)
    channels = ds.unique("channel")

    """ Channels are current hard-coded for fast testing """
    selected_channels = [channels[channels.index("@khalid_dinh")]]
    for channel in tqdm(selected_channels, desc="Processing channel: "):
      channel_audio_path = resolve_path(self.out_audio_path / channel)
      cur_channel_ds = filter_by_channel(ds=ds, channel=channel)
      splits = slice_dataset(ds=cur_channel_ds)

      """ Processing split to metadata """
      channel_metadata = self._process_channel_split(
        splits=splits,
        channel=channel,
        out_path=channel_audio_path
      )
      channel_metadata.to_csv(path_or_buf=channel_audio_path / "metadata.csv", index=False)

      """ Save the metadata to a HF dataset if needed """
      self._save_metadata_to_dataset(
        skip=False if save_to_dataset else True,
        metadata=channel_metadata,
        audio_path=channel_audio_path
      )

      self.logger.info(
        f"Saved {len(channel_metadata['file_name'])} audio files at {channel_audio_path}"
      )
      self.logger.info(f"Finished processing channel: {channel}")
