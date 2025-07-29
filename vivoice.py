import pandas as pd
import json
import soundfile as sf

from typing import Optional
from pathlib import Path
from utils.vivoice_utils import (
  load_vivoice,
  filter_by_channel,
  save_dataset,
  slice_dataset,
  merge_audio,
  process_text,
)
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
    """
    Initialize the preprocessor for the dataset capleaf/viVoice

    Args:
        out_audio_path (str): Path to save processed audios
        token (str): Hugging Face token
        dia_pipe (_type_): Pyannote's diarization pipeline
        asr_model (_type_): FasterWhisper model
    """
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

    # Step 1: Load dataset
    ds = load_vivoice(token=self.token)

    # Step 2: Filter dataset by channel
    index_dict = filter_by_channel(dataset=ds)

    # Step 3: Split dataset into valid lengths
    test_set = []
    for key, value in index_dict.items():
      if test_set and key not in test_set:
        continue

      self.logger.info(f"Current channel: {key}")

      channel_ds = ds.select(value)
      splits = slice_dataset(dataset=channel_ds)

      cur_channel = key
      metadata = pd.DataFrame(
        columns=["file_name", "channel", "text", "gen_text", "wer"]
      )
      out_channel_path = resolve_path(self.out_audio_path / cur_channel)

      # Step 4: Pass audio through pipeline
      for split in splits:
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
          self.logger.warning(
            f"Audio: {audio_name} has more than 1 potential speaker. Skipping saving audio."
          )
          continue

        split_text = split["text"]
        segments_text = []
        if not asr_res:
          self.logger.warning("ASR Result is empty.")
        else:
          for _, value in asr_res.items():
            if not value["segments"]:
              self.logger.warning("The current result segment is empty.")
              continue
            for seg in value["segments"]:
              segments_text.append(seg["text"])

        processed_text = process_text(
          references=split_text, predictions=segments_text
        )

        metadata.loc[len(metadata)] = {
          "file_name": audio_name,
          "channel": cur_channel,
          **processed_text,
        }

        sf.write(file=out_channel_path / audio_name, data=audio_waveform, samplerate=sample_rate)
        with open(out_channel_path / f"{audio_name[:-4]}.json", "w", encoding="utf-8") as outfile:
          json.dump(res, outfile, indent=4, ensure_ascii=False)

        self.logger.info(f"Finished processing {audio_name}")

      metadata.to_csv(out_channel_path / "metadata.csv", index=False)

      # Step 5: Save to dataset
      self.save_metadata_to_dataset(
        skip=False if save_to_dataset else True,
        metadata=metadata,
        audio_path=out_channel_path,
      )