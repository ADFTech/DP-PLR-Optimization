from functools import update_wrapper
from typing import TypeVar, Generic, Callable

# T is a type variable
T = TypeVar('T')

class Class_property(Generic[T]):
    """
    A decorator to create a class property.

    Args:
        method (Callable[..., T]): The method to be decorated.

    Returns:
        T: The value of the property.
    """
    def __init__(self, method: Callable[..., T]):
        """
        Initializes the decorator.
        """
        self.__method = method
        update_wrapper(self, method)

    def __get__(self, obj, cls=None) -> T:
        """
        Returns the value of the property.
        """
        if cls is None:
            cls = type(obj)
        return self.__method(cls)
