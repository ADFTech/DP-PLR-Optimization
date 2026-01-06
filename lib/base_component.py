from abc import ABC
from configparser import ConfigParser
from lib.logger import Logger, Omni_post
from lib.class_property import Class_property

class Base_component(ABC):
    """
    The base class for most components in the system. It provides a common
    interface for all components to access a shared configuration and logger.

    This class provides a framework for components that share a configuration
    parser and a logger. It ensures that each component can be configured and
    can log its activities in a standardized way.
    """

    def __init__(self, config: ConfigParser) -> None:
        """
        Initialize the Base component object with given configuration and logger.

        Args:
            config (ConfigParser): The configuration parser object.
        """
        self.config = config

        # Check if the configuration has the required sections and settings
        self.__validate_basic_config(self.config)

        # Use the local file logger if log_to_post_office is not set
        if not self.config[self.cls_name].get('log_to_post_office'):
            self.logger = Logger(
                log_name = self.cls_name,
                log_size = int(self.config['Logger']['log_size']),
                backup_count = int(self.config['Logger']['log_backup_count'])
            )
            return

        # If log_to_post_office is set, use the Post_office logger
        # Check if the Post_office credentials are set
        self.__valid_post_office_config(self.config)

        self.logger = Logger(
            log_name = self.cls_name,
            log_size = int(self.config['Logger']['log_size']),
            backup_count = int(self.config['Logger']['log_backup_count']),
            omni_post = Omni_post(
                url = self.config['Post_office']['url'],
                origin = self.config['Post_office']['origin'],
                username = self.config['Post_office']['username'],
                password = self.config['Post_office']['password'],
                queue = self.config['Post_office']['queue']
            )
        )

    # Getters and setters for the config
    @property
    def config(self) -> ConfigParser:
        """
        Gets the configuration parser object.
        """
        return self.__config

    @config.setter
    def config(self, config: ConfigParser) -> None:
        """
        Sets the configuration parser object.
        """
        if not isinstance(config, ConfigParser):
            raise ValueError("config must be an instance of ConfigParser.")
        self.__config = config

    # Getters and setters for the logger
    @property
    def logger(self) -> Logger:
        """
        Gets the logger object.
        """
        return self.__logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        """
        Sets the logger object.
        """
        if not isinstance(logger, Logger):
            raise ValueError("logger must be an instance of Logger.")
        self.__logger = logger

    @Class_property
    def cls_name(cls) -> str:
        """
        Shortcut to get the name of the component.
        """
        return cls.__name__

    def __validate_basic_config(self, config: ConfigParser) -> None:
        """
        Validates the configuration settings for the component.
        """
        if not config.has_section(self.cls_name):
            raise ValueError(f"Configuration file does not have a section for {self.cls_name}.")
        if not config.has_section('Logger'):
            raise ValueError("Configuration file does not have a section for Logger.")
        if not all (k in config['Logger'] for k in ('log_size', 'log_backup_count')):
            raise ValueError("Logger settings are not set in the configuration file.")

    def __valid_post_office_config(self, config: ConfigParser) -> None:
        """
        Validates the Post_office credentials in the configuration.
        """
        if not config.has_section('Post_office'):
            raise ValueError("Configuration file does not have a section for Post_office.")
        if not all (k in config['Post_office'] for k in ('origin', 'username', 'password', 'queue')):
            raise ValueError("Post_office credentials are not set in the configuration file.")
