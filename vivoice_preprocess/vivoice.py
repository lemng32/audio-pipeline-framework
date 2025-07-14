import pandas as pd

from tqdm import tqdm
from typing import Optional
from pathlib import Path
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


class VivoicePreprocessor:
  def __init__(
    self,
    out_audio_path,
    token,
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
  ):
    if save_to_dataset:
      self.out_dataset_path = resolve_path(out_dataset_path)

    # Step 1: Load dataset
    ds = load_vivoice(token=self.token)

    # Step 2: Filter dataset by channel
    index_dict = filter_by_channel(dataset=ds)

    # Step 3: Split dataset into valid lengths
    test_channel = {"@khalid_dinh": index_dict["@khalid_dinh"]}
    for key, value in test_channel.items():
      channel_ds = ds.select(value)
      splits = slice_dataset(dataset=channel_ds)

      cur_channel = key
      metadata = pd.DataFrame(
        columns=["file_name", "channel", "text", "gen_text", "wer"]
      )
      out_channel_path = resolve_path(self.out_audio_path / cur_channel)

      # Step 4: Process with pipeline
      for split in splits:
        texts = split["text"]
        audios = [audio["array"] for audio in split["audio"]]
        sample_rate = split["audio"][0]["sampling_rate"]

        merged_audio = merge_audio(audios=audios)
        res = self.pipeline(merged_audio, sample_rate)

        speakers = set(res["diarize_df"]["speaker"])
        if len(speakers) > 1:
          continue

        segments = res["segments"]
        segments_text = [segment.text for segment in segments]

        # TO-DO: Add some metric to compare the original text to the generated text

        metadata.loc[len(metadata)] = {
          "file_name": res["audio"]["name"],
          "channel": cur_channel,
          "text": " ".join(texts),
          "gen_text": "".join(segments_text),
        }

        save_audio(
          out_file=out_channel_path / f"{res['audio']['name']}.wav",
          audio=res["audio"]["waveform"],
        )

      metadata.to_csv(out_channel_path / "metadata.csv", index=False)

      # Step 5: Save to dataset
      self.save_metadata_to_dataset(
        skip=False if save_to_dataset else True,
        metadata=metadata,
        audio_path=out_channel_path,
      )
