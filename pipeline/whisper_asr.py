import torch
import numpy as np

from utils.logger import get_logger
from faster_whisper import WhisperModel, BatchedInferencePipeline


class FasterWhisperASR:
  """
  Wrapper class for WhisperModel from faster-whisper
  """

  def __init__(
    self,
    model_size: str = "large-v3",
    compute_type: str = "float16",
    transcribe_options: dict = None,
  ) -> None:
    """
    Initialize the model instance

    Args:
      model_size (str, optional): The whisper model to use. Defaults to "large-v3".
      batch (bool, optional): Whether to enable batch inference. Defaults to False.
      compute_type (str, optional): Data precision for inference. Defaults to "float16".
      transcribe_options (dict, optional): Options to pass to the transcribe method. Defaults to None.
    """
    self.logger = get_logger(__name__)
    self.model = self.load_model(model_size=model_size, compute_type=compute_type)
    self.transcribe_options = transcribe_options

  def get_gpu(self) -> tuple:
    """
    Detect available GPU devices.

    Returns:
      tuple: (device, device_index)
        - device (str): "cuda" if GPU is available, otherwise "cpu".
        - device_index (int or list[int]): GPU indices or 0 if using CPU.
    """
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

  def load_model(self, model_size: str, compute_type: str) -> BatchedInferencePipeline:
    """
    Load the whisper model.

    Args:
      model_size (str): The whisper model to use.
      compute_type (str): Data precision for inference.

    Returns:
      object: Loaded model instance.
    """
    device, device_index = self.get_gpu()
    model = WhisperModel(
      model_size, device=device, device_index=device_index, compute_type=compute_type
    )
    model = BatchedInferencePipeline(model)
    return model

  def transcribe_audio(self, audio: np.ndarray) -> tuple:
    """
    Transcribe audio given in waveform

    Args:
      audio (np.ndarray): 1D NumPy array of float32.

    Returns:
      tuple: The audio segments and transcription info
    """
    segments, info = self.model.transcribe(
      audio=audio, **(self.transcribe_options or {})
    )

    return (segments, info)
