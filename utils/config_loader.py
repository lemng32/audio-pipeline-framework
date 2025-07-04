# utils/config_loader.py
import json
from pathlib import Path


class ConfigLoader:
  def __init__(self, path: str = "config.json"):
    self.config_path = Path(path)
    self._config = self._load_config()
    self._apply_defaults()

  def _load_config(self) -> dict:
    if not self.config_path.exists():
      raise FileNotFoundError(f"Config file not found: {self.config_path}")

    with self.config_path.open("r") as f:
      try:
        return json.load(f)
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in config file: {e}")

  def _apply_defaults(self):
    if self._config.get("out_audio_path", "") == "":
      self._config["out_audio_path"] = self._config["default_audio_path"]

    if self._config.get("out_dataset_path", "") == "":
      self._config["out_dataset_path"] = self._config["default_dataset_path"]

  @property
  def config(self) -> dict:
    return self._config
