from abc import abstractmethod
from configparser import ConfigParser
from lib.base_component import Base_component

class Executable(Base_component):
    """
    Abstract base class for creating executable objects with a common configuration and logging.

    This class provides a framework for executable components that share a configuration
    parser and a logger. It ensures that each executable can be configured and can log
    its activities in a standardized way.
    """
    def __init__(self, config:ConfigParser) -> None:
        """
        Initialize the Executable object with given configuration and logger.

        Args:
            config (ConfigParser): The configuration parser object.
        """
        super().__init__(config)

    @abstractmethod
    def execute(self) -> any:
        """
        Abstract execute method to execute arbitrary code.

        This method must be overridden by subclasses to provide specific
        executable behavior. The return value can be of any type as determined by
        the subclass implementation.

        Returns:
            The return value is dynamic (denoted by 'any') and will vary based
            on the subclass's implementation.
        """
        raise NotImplementedError("execute() must be implemented by subclass.")
