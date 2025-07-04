# utils/logger.py
import logging
import sys


def get_logger(name: str = "app") -> logging.Logger:
  logger = logging.getLogger(name)

  if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
      "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  return logger
