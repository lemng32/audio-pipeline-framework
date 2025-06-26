from pathlib import Path


def resolve_path(path_str: str) -> str:
  if not dir:
    raise ValueError("Directory cannot be of value: None")
  path = Path(path_str).expanduser().resolve()
  if not path.is_dir():
    path.mkdir(parents=True, exist_ok=True)
  return str(path)