# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import logging
import sys
from pathlib import Path
from time import strftime
from typing import Any, no_type_check

ALL_LOGS = ROOT / "logs"
LOGS = ALL_LOGS / "debug"
if not LOGS.exists():
    LOGS.mkdir(exist_ok=True, parents=True)
LOGGER_NAME = "logger"  # just the logger name to use with logging.getLogger


def setup_logging() -> None:
    """Must be called first before all other code."""
    timestamp = strftime("%Y-%b-%d__%H-%M-%S")
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(LOGS / f"{timestamp}.log"))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(levelname)s (%(asctime)s) [pid:%(process)d]  %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def func_with_name() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")


# https://stackoverflow.com/q/6760685
class Singleton(type):
    _instances = {}  # type: ignore

    @no_type_check
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    """Bypass annoying verbosity of `logging.getLogger(NAME)` by creating a singleton."""

    def __init__(self) -> None:
        setup_logging()

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """Log with lowest priority. For most detailed messages that we do NOT want to see
        in stdout / stderr but still have logged in the log file."""
        logging.getLogger(LOGGER_NAME).debug(*args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Log with second-lowest priority. DOES show up in stdout / stderr and log files,
        so probabably mostly this is the one you want when you just want to see what the program is
        doing. Should not be used for errors or problems."""
        logging.getLogger(LOGGER_NAME).info(*args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        """Log with priority WARNING (more than DEBUG and INFO, less than ERROR and CRITICAL).
        Use for things we should probably fix eventually, i.e. problematic code, but which
        are not worth terminating the program."""
        logging.getLogger(LOGGER_NAME).warning(*args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        """Log an actual error that is not properly handled, but from which we can recover. Most of
        the time if you are using this, you should be dumping a traceback here, e.g.
        LOGGER.error(traceback.format_exc()). If the error cannot be recovered from (further code
        will produce broken or useless results) then you should instead use LOGGER.critical and then
        terminate the program."""
        logging.getLogger(LOGGER_NAME).error(*args, **kwargs)

    def critical(self, *args: Any, **kwargs: Any) -> None:
        """Log an error or problem that requires terminating execution. Also try to exit
        gracefully after if possible."""
        logging.getLogger(LOGGER_NAME).critical(*args, **kwargs)


LOGGER = Logger()

if __name__ == "__main__":
    LOGGER.error("No setup needed!")
    LOGGER.debug("Won't see me in the console!")
