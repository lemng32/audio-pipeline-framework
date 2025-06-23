import json

from vivoice_preprocess import vivoice_preprocess

with open("config.json", "r") as f:
  conf = json.load(f)

vivoice_preprocess.preprocess(
  token=conf["huggingface_token"],
  out_audio_path=conf["out_audio_path"],
  load_from_disk=True,
  dataset_disk_path=conf["dataset_path"]
)