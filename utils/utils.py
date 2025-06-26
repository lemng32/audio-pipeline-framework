from pathlib import Path


def resolve_path(path_str: str) -> str:
  """
  Ensure that the provided path is well defined and create if it does not exist.
  
  Args:
      path_str (str): The path string.

  Raises:
      ValueError: Directory cannot be of value: None.

  Returns:
      str: The resolved path.
  """
  if not dir:
    raise ValueError("Directory cannot be of value: None")
  path = Path(path_str).expanduser().resolve()
  if not path.is_dir():
    path.mkdir(parents=True, exist_ok=True)
  return str(path)