import json

from vivoice_preprocess import vivoice_preprocess


def main_process():
  vivoice_preprocess.preprocess(
    out_audio_path=out_audio_path,
    token=conf["huggingface_token"],
    load_from_disk=True,
    dataset_disk_path=conf["dataset_path"],
    diarize_filter=False,
  )


if __name__ == "__main__":
  with open("config.json", "r") as f:
    try:
      conf = json.load(f)
    except json.decoder.JSONDecodeError as _:
      raise TypeError("sybau")

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
