from vivoice_preprocess.preprocessor import VivoicePreprocessor
from utils.config_loader import ConfigLoader


def main_process(config: dict):
  """
  TO-DO: Add description
  """
  preprocessor = VivoicePreprocessor(
    out_audio_path=config["out_audio_path"],
    token=config["huggingface_token"]
  )
  preprocessor.run(save_to_dataset=True, out_dataset_path=config["out_dataset_path"])


if __name__ == "__main__":
  loader = ConfigLoader(path="config.json")
  conf = loader.config
  
  main_process(conf)
