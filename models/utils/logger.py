import sys
from loguru import logger

def setup_logger():
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    logger.remove()
    logger.add(sys.stdout, format=log_format, level="INFO")
    logger.add("logs/app.log", rotation="10 MB", format=log_format, level="DEBUG")
    logger.info("Logger initialized.")

setup_logger()