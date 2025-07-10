from vivoice_preprocess.vivoice import VivoicePreprocessor
from utils.config_loader import ConfigLoader


def main_process(config: dict):
  """
  TO-DO: Add description
  """
  preprocessor = VivoicePreprocessor(
    out_audio_path=config["out_audio_path"],
    token=config["huggingface_token"]
  )
  preprocessor.run()


if __name__ == "__main__":
  loader = ConfigLoader(path="config_local.json")
  conf = loader.config
  
  main_process(conf)
