from sys import stdout

from loguru import logger

BOT_TOKEN = "7878286640:AAH12xKQwFlvWKhybxmxVnaBuKIkLvvxBdc"
USE_GPU = False  # True если используется Cuda


def setup_logger():
    logger.remove()
    logger.add(
        stdout,
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
        level="DEBUG",
    )
