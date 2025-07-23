import torch

from tqdm.contrib.logging import logging_redirect_tqdm
from pyannote.audio import Pipeline
from vivoice import VivoicePreprocessor
from pipeline.whisper_asr import FasterWhisperASR
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


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

if __name__ == "__main__":
  logger = get_logger(__name__)

  with logging_redirect_tqdm(loggers=[logger]): 
    loader = ConfigLoader(path="config.json")
    conf = loader.config

    dia_pipe = pipe = Pipeline.from_pretrained(
      "pyannote/speaker-diarization-3.1",
      use_auth_token=conf["huggingface_token"],
    )
    dia_pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Diariziation pipeline created.")

    asr_model = FasterWhisperASR(
      model_size=conf["model_size"],
      compute_type=conf["compute_type"],
      transcribe_options=conf["transcribe_options"]
    )
    logger.info("Whisper model created.")

    main_process()
