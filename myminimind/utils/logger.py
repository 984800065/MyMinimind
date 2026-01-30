import os
import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | "
    "<cyan>{file.path}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{function}</magenta> | "
    "<level>{message}</level>"
)

def setup_logger():
    # 移除默认logger
    logger.remove()
    rank = int(os.environ.get("RANK", 0))

    # 控制台输出（彩色）
    if rank == 0:
        logger.add(
            sys.stdout,
            format=LOG_FORMAT,
            level="DEBUG",
            colorize=True,
            enqueue=True
        )

        # 文件输出（无颜色）
        logger.add(
            LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {file.path}:{line} | {function} | {message}",
            level="DEBUG",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    return logger

logger = setup_logger()

# 示例
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")