import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO):

    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = RotatingFileHandler(
        log_dir / log_file, maxBytes=10485760, backupCount=5
    )
    console_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create main logger
logger = setup_logger('anomaly_detection', 'anomaly_detection.log')