import os
import atexit
import logging
import requests
import traceback
import concurrent.futures
from threading import Lock
from datetime import datetime
from pydantic import BaseModel, AnyHttpUrl
from logging.handlers import RotatingFileHandler

class Executor():
    __executor = None
    _lock = Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls.__executor is None:
                cls.__executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            return cls.__executor

    @classmethod
    def shutdown(cls, **kwargs):
        with cls._lock:
            if cls.__executor is not None:
                cls.__executor.shutdown(wait=True)
                cls.__executor = None


class Log_schema_v1(BaseModel):
    origin: str
    name: str
    level: int
    message: str
    traceback: str | None = None
    timestamp: datetime

    @property
    def VERSION(self) -> str:
        """
        Returns the version of the logger.
        """
        return "Log_schema_v1"

    @property
    def dict(self) -> dict:
        return {
            "Header": {
                "Version": self.VERSION
            },
            "Body": {
                "origin": self.origin,
                "name": self.name,
                "level": self.level,
                "message": self.message,
                "traceback": self.traceback,
                "timestamp": self.timestamp.isoformat()
            }
        }


class Omni_post(BaseModel):
    url: AnyHttpUrl
    origin: str
    username: str
    password: str
    queue: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__executor = Executor.get()
        atexit.register(self.__shutdown)

    def log(self, log: Log_schema_v1) -> None:
        """
        Sends a log message to Omni post.
        """
        if not isinstance(log, Log_schema_v1):
            raise ValueError("Log message must be a Log_schema_v1.")

        # Add the RMQ credentials to the header
        message = log.dict
        message['Header'] = {
            "Version": message['Header']['Version'],
            "RMQ_username": self.username,
            "RMQ_password": self.password,
            "RMQ_queue": self.queue
        }
        self.__executor.submit(self.__send_request, message)

    def __send_request(self, message: dict) -> None:
        """
        Sends the log message to Omni post.
        """
        try:
            response = requests.post(
                url = self.url,
                json = message
            )
        except Exception as error:
            print(f"Error sending log message to Omni post. | {error}")

        else:
            if response.status_code != 200:
                print(f"Error sending log message to Omni post. | {response.status_code} | {response.text}")

    def __shutdown(self):
        self.__executor.shutdown(wait=True)

class Logger():
    """
    Custom logging class for creating a rotating file logger.

    This logger supports rotation of the log files when they reach a certain size and keeps a specified number of backup copies.
    Logging can be done at various levels including debug, info, warning, error, and critical.

    It also supports auxiliary logging to Omni post if the Omni_post object is provided.
    """
    LOG_FORMAT = logging.Formatter('[%(levelname)s][%(asctime)s][%(name)s] %(message)s')

    def __init__(self,
        log_name: str,
        log_size: int,
        backup_count: int,
        level: int = logging.INFO,
        omni_post: Omni_post = None,
    ) -> None:
        self.log_file = log_name
        self.log_name = log_name
        self.log_size = log_size
        self.backup_count = backup_count
        self.level = level
        self.omni_post = omni_post
        self.__logger = None
        self.init_rotating_file_handler()
        self.init_logger()

    # Getters and setters for log_file
    @property
    def log_file(self) -> str:
        """
        Gets the path to the log file.
        """
        return self.__log_file

    @log_file.setter
    def log_file(self, value: str) -> None:
        """
        Sets the path to the log file. Creates 'logs' directory if it doesn't exist.
        """
        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.__log_file = f'logs/{value}.log'

    # Getters and setters for log_name
    @property
    def log_name(self) -> str:
        """
        Gets the name of the logger.
        """
        return self.__log_name

    @log_name.setter
    def log_name(self, value: str) -> None:
        """
        Sets the name of the logger.
        """
        self.__log_name = value

    # Getters and setters for log_size
    @property
    def log_size(self) -> int:
        """
        Gets the log size in bytes.
        """
        return self.__log_size

    @log_size.setter
    def log_size(self, value: int) -> None:
        """
        Sets the maximum size of the log file in megabytes before rotation.
        """
        self.__log_size = value * 1024 * 1024  # Convert MB to bytes

    # Getters and setters for backup_count
    @property
    def backup_count(self) -> int:
        """
        Gets the backup count.
        """
        return self.__backup_count

    @backup_count.setter
    def backup_count(self, value: int) -> None:
        """
        Sets the number of backup log files to keep.
        """
        self.__backup_count = value

    # Getters and setters for level
    @property
    def level(self) -> int:
        """
        Gets the logging level.
        """
        return self.__level

    @level.setter
    def level(self, value: int) -> None:
        """
        Sets the logging level.
        """
        self.__level = value

    # Getters and setters for omni_post
    @property
    def omni_post(self) -> Omni_post:
        """
        Gets the Omni_post object.
        """
        return self.__omni_post

    @omni_post.setter
    def omni_post(self, value: Omni_post) -> None:
        """
        Sets the Omni_post object.
        """
        if value is not None and not isinstance(value, Omni_post):
            raise ValueError("omni_post must be an Omni_post object.")
        self.__omni_post = value

    def init_rotating_file_handler(self) -> None:
        """
        Initializes the rotating file handler for the logger.
        """
        rotating_file_handler = RotatingFileHandler(
            self.log_file,
            mode='a',
            maxBytes=self.log_size,
            backupCount=self.backup_count,
            encoding=None
        )
        rotating_file_handler.setFormatter(self.LOG_FORMAT)
        rotating_file_handler.setLevel(self.level)
        self.__rotating_file_handler = rotating_file_handler

    def init_logger(self) -> None:
        """
        Initializes the logger with the rotating file handler.
        """
        logger = logging.getLogger(self.log_name)
        logger.setLevel(self.level)
        if not logger.handlers:
            logger.addHandler(self.__rotating_file_handler)
        self.__logger = logger

    def log(self, message:str, level: int, traceback: str = None, prints: bool = False) -> None:
        """
        Generic log function.
        """
        if not isinstance(message, str):
            raise ValueError("Log message must be a string.")

        if not isinstance(level, int):
            raise ValueError("Log level must be an integer.")

        if traceback is not None and not isinstance(traceback, str):
            raise ValueError("Traceback must be a string.")

        # Log the message
        if traceback:
            # If there is a traceback, log it as well
            self.__logger.log(level, f"{message}\n{traceback}")
            if prints: print(f"{message}\n{traceback}")
        else:
            # Otherwise, just log the message
            self.__logger.log(level, message)
            if prints: print(message)


        if not self.omni_post:
            return

        self.omni_post.log(
            Log_schema_v1(
                origin = self.omni_post.origin,
                name = self.log_name,
                level = level,
                message = message,
                traceback = traceback,
                timestamp = datetime.utcnow()
            )
        )

    def log_debug(self, message:str, prints: bool = False) -> None:
        """
        Logs a debug message.
        """
        self.log(message, logging.DEBUG, prints = prints)

    def log_info(self, message:str, prints: bool = False) -> None:
        """
        Logs an info message.
        """
        self.log(message, logging.INFO, prints = prints)

    def log_warning(self, message:str, prints: bool = False) -> None:
        """
        Logs a warning message.
        """
        self.log(message, logging.WARNING, prints = prints)

    def log_critical(self, message:str, prints: bool = False) -> None:
        """
        Logs a critical message.
        """
        self.log(message, logging.CRITICAL, prints = prints)

    def log_error(self, message:str, prints: bool = False) -> None:
        """
        Logs an error message and includes the traceback.
        """
        traceback_str = traceback.format_exc()
        if "NoneType: None" in traceback_str:
            self.log(message, logging.ERROR, prints = prints)
        else:
            self.log(message, logging.ERROR, traceback = traceback_str, prints = prints)
