import pandas as pd

from tqdm import tqdm
from typing import Optional
from pathlib import Path
from underthesea import text_normalize
from evaluate import load
from utils.dataset_utils import (
  load_vivoice,
  filter_by_channel,
  save_dataset,
  slice_dataset,
)
from utils.audio_utils import merge_audio, save_audio
from utils.file_utils import resolve_path
from utils.logger import get_logger
from vivoice_preprocess.preprocessor import PreprocessorPipeline

import os

class VivoicePreprocessor:
  """
  Preprocessor for the capleaf/viVoice dataset on Hugging Face
  """

  def __init__(
    self,
    out_audio_path: str,
    token: str,
    dia_pipe,
    asr_model,
  ) -> None:
    self.logger = get_logger(__name__)
    self.out_audio_path = resolve_path(out_audio_path)
    self.out_dataset_path = None
    self.token = token

    self.pipeline = PreprocessorPipeline(dia_pipe=dia_pipe, asr_model=asr_model)

  def save_metadata_to_dataset(
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
    Run the preprocessor

    Args:
        save_to_dataset (Optional[bool], optional): Whether to save audio as a Hugging Face dataset. Defaults to False.
        out_dataset_path (Optional[str], optional): Where to store the saved dataset if enabled. Defaults to None.
    """
    if save_to_dataset:
      self.out_dataset_path = resolve_path(out_dataset_path)

    metric = load("wer")

    # Step 1: Load dataset
    ds = load_vivoice(token=self.token)

    # Step 2: Filter dataset by channel
    index_dict = filter_by_channel(dataset=ds)

    # Step 3: Split dataset into valid lengths
    # Hard-coded channels for quick run
    test_channels = {
      "@khalid_dinh": index_dict["@khalid_dinh"],
      # "@zombiev4": index_dict["@zombiev4"],
    }
    for key, value in tqdm(test_channels.items(), desc="Processing channels: "):
      # for key, value in tqdm(index_dict.items(), desc="Processing channels: "):
      tqdm.write(f"Current channel: {key}")

      channel_ds = ds.select(value)
      splits = slice_dataset(dataset=channel_ds)

      cur_channel = key
      metadata = pd.DataFrame(
        columns=["file_name", "channel", "text", "gen_text", "wer"]
      )
      out_channel_path = resolve_path(self.out_audio_path / cur_channel)

      # Step 4: Process with pipeline
      for split in tqdm(splits, desc="Processing channel split: "):
        split_texts = split["text"]
        audios = [audio["array"] for audio in split["audio"]]
        sample_rate = split["audio"][0]["sampling_rate"]

        merged_audio = merge_audio(audios=audios)
        res = self.pipeline(merged_audio, sample_rate)

        speakers = set(res["diarize_df"]["speaker"])
        if len(speakers) > 1:
          self.logger.info(
            f"Number of potential speakers: {len(speakers)}. Skipping audio."
          )
          continue

        segments = res["segments"]
        segments_text = [segment.text for segment in segments]

        metadata_text = " ".join(split_texts)
        metadata_gen_text = "".join(segments_text)

        norm_reference = text_normalize(metadata_text)
        norm_prediction = text_normalize(metadata_gen_text)
        wer = metric.compute(references=[norm_reference], predictions=[norm_prediction])

        metadata.loc[len(metadata)] = {
          "file_name": res["audio"]["name"],
          "channel": cur_channel,
          "text": metadata_text,
          "gen_text": metadata_gen_text,
          "wer": wer,
        }

        save_audio(
          out_file=out_channel_path / f"{res['audio']['name']}.wav",
          audio=res["audio"]["waveform"],
        )

        tqdm.write(f"Finished processing {res['audio']['name']}")

      metadata.to_csv(out_channel_path / "metadata.csv", index=False)

      # Step 5: Save to dataset
      self.save_metadata_to_dataset(
        skip=False if save_to_dataset else True,
        metadata=metadata,
        audio_path=out_channel_path,
      )