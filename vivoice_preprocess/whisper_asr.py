import torch

from typing import Optional, Iterable
from utils.logger import get_logger
from faster_whisper import WhisperModel, BatchedInferencePipeline


class FasterWhisperASR:
  def __init__(
    self,
    model_size: str = "large-v3",
    batch: bool = False,
    compute_type: str = "float16"
  ):
    self.logger = get_logger(__name__)
    self.model = self.load_model(model_size=model_size, batch=batch, compute_type=compute_type)

  def get_gpu(self):
    device = "cpu"
    device_index = 0

    if torch.cuda.is_available():
      self.logger.info("CUDA is available.")
      if torch.backends.cudnn.is_available():
        self.logger.info(f"cuDNN Enabled. Version: {torch.backends.cudnn.version()}")
        num_gpus = torch.cuda.device_count()
        device = "cuda"
        device_index = list(range(num_gpus))
      else:
        self.logger.info("cuDNN is not enabled. Falling back to CPU.")
    else:
      self.logger.info("CUDA is not available. Using CPU.")

    return (device, device_index)

  def load_model(self, model_size: str, batch: bool, compute_type: str):
    device, device_index = self.get_gpu()
    model = WhisperModel(
      model_size,
      device=device,
      device_index=device_index,
      compute_type=compute_type
    )
    if batch:
      self.logger.info("Using batched model")
      model = BatchedInferencePipeline(model)
    return model
  
  def transcribe_audio(
    self,
    audio,
    beam_size = 5,
    condition_on_previous_text = True,
    without_timestamps = False,
    vad_filter: bool = True,
    language: Optional[str] = None,
    batch_size: Optional[int] = None,
  ) -> Iterable:
    options = {
      "audio": audio,
      "beam_size": beam_size,
      "condition_on_previous_text": condition_on_previous_text,
      "without_timestamps": without_timestamps,
      "vad_filter": vad_filter,
    }

    if language:
      options["language"] = language  

    if isinstance(self.model, BatchedInferencePipeline):
      if not batch_size:
        self.logger.warning("Using batched model but batch size not provided, using default of 8")
        options["batch_size"] = 8
    else:
      if batch_size:
        self.logger.warning(f"Not using batched model but batch size of {batch_size} is provided")

    segments, info = self.model.transcribe(**options)
    self.logger.info("Finished processing audio")

    return (segments, info)
