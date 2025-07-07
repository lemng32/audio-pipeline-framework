import os

from pathlib import Path
from contextlib import contextmanager


def resolve_path(path_str: str) -> Path:
  """
  Convert a string path into a resolved Path object.
  If the directory does not exist, it is created.

  Args:
    path_str (str): The path as a string (can include ~).

  Returns:
    Path: A fully resolved directory Path.

  Raises:
    ValueError: If the path string is empty or None.
  """
  if not dir:
    raise ValueError("Directory cannot be of value: None")
  path = Path(path_str).expanduser().resolve()
  if not path.is_dir():
    path.mkdir(parents=True, exist_ok=True)
  return path


@contextmanager
def change_working_dir(path: Path):
  """
  Context manager to temporarily change the working directory.

  Args:
    path (Path): Directory to switch into temporarily.

  Yields:
    None: Executes code in the context of the new working directory.

  Example:
    with change_working_dir(Path("/tmp")):
      # do something in /tmp
  """
  prev_cwd = os.getcwd()
  os.chdir(path=path)
  try:
    yield
  finally:
    os.chdir(path=prev_cwd)
