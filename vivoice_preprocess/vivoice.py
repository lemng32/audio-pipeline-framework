import pandas as pd

from tqdm import tqdm
from evaluate import load
from utils.dataset_utils import load_vivoice, slice_dataset
from utils.audio_utils import merge_audio, save_audio
from utils.file_utils import resolve_path
from utils.logger import get_logger
from vivoice_preprocess.preprocessor import PreprocessorPipeline

class VivoicePreprocessor:
  def __init__(
    self, out_audio_path: str, token: str, use_diarization: bool = True
  ) -> None:
    self.out_audio_path = resolve_path(out_audio_path)
    self.out_dataset_path = None
    self.token = token
    self.metric = load("wer")
    self.logger = get_logger(__name__)

    self.pipeline = PreprocessorPipeline(self.token)
  
  def run(self):
    # Step 1: Load dataset
    ds = load_vivoice(token=self.token)
    # Step 2: Filter dataset by channel
    channels = ds.unique(column="channel")
    index_dict = { channel: [] for channel in channels }
    for i, channel in tqdm(enumerate(ds["channel"])):
      index_dict[channel].append(i)
    # Step 3: Split dataset into valid lengths
    test_channel = { "@khalid_dinh": index_dict["@khalid_dinh"] }
    for key, value in test_channel.items():
      cur_channel = key
      channel_ds = ds.select(value)
      metadata = pd.DataFrame(columns=["file_name", "channel", "text", "gen_text", "wer"])
      out_channel_path = resolve_path(self.out_audio_path / cur_channel)
      splits = slice_dataset(dataset=channel_ds)
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
        
        segments, _ = self.pipeline.whisper_asr(res["audio"])
        segments_text = [segment.text for segment in segments]

        wer = self.metric.compute(predictions=[" ".join(segments_text)], references=[" ".join(texts)])

        metadata.loc[len(metadata)] = {
          "file_name": res["audio"]["name"],
          "channel": cur_channel,
          "text": " ".join(texts),
          "gen_text": " ".join(segments_text),
          "wer": wer
        }

        save_audio(out_file=out_channel_path / f"{res["audio"]["name"]}.wav", audio=res["audio"]["waveform"])
      
      metadata.to_csv(out_channel_path / "metadata.csv", index=False)
