import logging
log_level = logging.INFO
logger = logging.getLogger("HistoMIL")
# check if logger has been initialized
if not logger.hasHandlers():
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
