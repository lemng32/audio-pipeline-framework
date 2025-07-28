import pandas as pd
import re
import json

from tqdm import tqdm
from typing import Optional
from pathlib import Path
from underthesea import text_normalize
from evaluate import load
from num2words import num2words
from utils.dataset_utils import (
  load_vivoice,
  filter_by_channel,
  save_dataset,
  slice_dataset,
)
from utils.audio_utils import merge_audio, save_audio
from utils.file_utils import resolve_path
from utils.logger import get_logger
from pipeline.preprocessor import PreprocessorPipeline


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
    self.metric = load("wer")

    self.pipeline = PreprocessorPipeline(dia_pipe=dia_pipe, asr_model=asr_model)

  def process_text(self, references: list, predictions: list) -> dict:
    """
    Process the given reference and prediction text.

    Args:
      references (list): The list of reference text.
      predictions (list): The list of ASR genererated text.

    Returns:
      dict: The dictionary to be added to the metadata, containing:
        {
          "text": The normalized reference,
          "gen_text": The normalized predictions,
          "wer": The WER score
        }
    """
    
    # Helper function to normalize text
    def normalize_helper(text) -> str:
      # Remove punctuation (keep numbers and letters)
      text = re.sub(r'[^\w\s]', '', text)

      # Normalize whitespace
      text = re.sub(r"\s+", " ", text).strip()

      # Convert number to text
      text = re.sub(r"\b\d+\b", lambda m: num2words(int(m.group()), lang="vi"), text)

      norm_text = text_normalize(text)
      norm_text = norm_text.lower()
      return norm_text

    metadata_text = " ".join(references)
    metadata_gen_text = " ".join(predictions)

    norm_reference = normalize_helper(metadata_text)
    norm_prediction = normalize_helper(metadata_gen_text)

    wer = self.metric.compute(
      references=[norm_reference], predictions=[norm_prediction]
    )

    return {"text": norm_reference, "gen_text": norm_prediction, "wer": wer}

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

    # Step 1: Load dataset
    ds = load_vivoice(token=self.token)

    # Step 2: Filter dataset by channel
    index_dict = filter_by_channel(dataset=ds)

    # Step 3: Split dataset into valid lengths
    for key, value in tqdm(index_dict.items(), desc="Processing channels: "):
      self.logger.info(f"Current channel: {key}")

      channel_ds = ds.select(value)
      splits = slice_dataset(dataset=channel_ds)

      cur_channel = key
      metadata = pd.DataFrame(
        columns=["file_name", "channel", "text", "gen_text", "wer"]
      )
      out_channel_path = resolve_path(self.out_audio_path / cur_channel)

      # Step 4: Process with pipeline
      for split in tqdm(splits, desc="Processing channel split: "):
        audios = [audio["array"] for audio in split["audio"]]
        sample_rate = split["audio"][0]["sampling_rate"]

        merged_audio = merge_audio(audios=audios)

        audio, res = self.pipeline(audio=merged_audio, sample_rate=sample_rate)

        audio_name = audio["name"]
        audio_waveform = audio["waveform"]
        diarize_res = res["diarize_res"]
        asr_res = res["asr_res"]

        speakers = set([row["speaker"] for row in diarize_res])
        if len(speakers) > 1:
          tqdm.write(
            f"Audio: {audio_name} has more than 1 potential speaker. Skipping saving audio."
          )
          continue

        split_text = split["text"]
        segments_text = []
        if not asr_res:
          self.logger.warning("ASR Result is empty.")
        else:
          for _, res in asr_res.items():
            if not res["segments"]:
              self.logger.warning("The current result segment is empty.")
              continue
            for seg in res["segments"]:
              segments_text.append(seg["text"])

        processed_text = self.process_text(
          references=split_text, predictions=segments_text
        )

        metadata.loc[len(metadata)] = {
          "file_name": audio_name,
          "channel": cur_channel,
          **processed_text,
        }

        save_audio(
          out_file=out_channel_path / audio_name,
          audio=audio_waveform,
        )
        with open(out_channel_path / f"{audio_name}.json", "w", encoding="utf-8") as outfile:
          json.dump(res, outfile, indent=4, ensure_ascii=False)

        self.logger.info(f"Finished processing {audio_name}")

      metadata.to_csv(out_channel_path / "metadata.csv", index=False)

      # Step 5: Save to dataset
      self.save_metadata_to_dataset(
        skip=False if save_to_dataset else True,
        metadata=metadata,
        audio_path=out_channel_path,
      )