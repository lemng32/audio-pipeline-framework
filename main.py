import json
from vivoice_preprocess.preprocessor import VivoicePreprocessor


def main_process():
  """
  TO-DO: Add description
  """
  preprocessor = VivoicePreprocessor(
    out_audio_path=out_audio_path, token=conf["huggingface_token"]
  )
  preprocessor.run(save_to_dataset=True, out_dataset_path=out_dataset_path)


if __name__ == "__main__":
  with open("config_local.json", "r") as f:
    try:
      conf = json.load(f)
    except json.decoder.JSONDecodeError as _:
      raise TypeError("Missing key in configuration file")

  # Use config's default path value if none is provided
  out_audio_path = (
    conf["default_audio_path"]
    if conf["out_audio_path"] == ""
    else conf["out_audio_path"]
  )
  out_dataset_path = (
    conf["default_dataset_path"]
    if conf["out_dataset_path"] == ""
    else conf["out_dataset_path"]
  )

  main_process()
