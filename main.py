import torch

from pyannote.audio import Pipeline
from vivoice_preprocess.vivoice import VivoicePreprocessor
from vivoice_preprocess.whisper_asr import FasterWhisperASR
from utils.config_loader import ConfigLoader


def main_process():
  """
  TO-DO: Add description
  """
  vivoice_preprocessor = VivoicePreprocessor(
    out_audio_path=conf["out_audio_path"],
    token=conf["huggingface_token"],
    dia_pipe=dia_pipe,
    asr_model=asr_model,
  )
  vivoice_preprocessor.run()
  # vivoice_preprocessor.test_pipeline("D:/Stuff/emandai/processed_audio/@khalid_dinh")

if __name__ == "__main__":
  loader = ConfigLoader(path="config.json")
  conf = loader.config

  dia_pipe = pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=conf["huggingface_token"],
  )
  dia_pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
  
  asr_model = FasterWhisperASR(model_size="turbo")

  main_process()
