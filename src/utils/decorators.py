import contextlib
import functools

from .logger import print_error


class TryExcept(contextlib.ContextDecorator):
    """Usage: @TryExcept() decorator or 'with TryExcept():' context manager."""

    def __init__(self, msg: str = "", verbose: bool = True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __call__(self, func):
        """
        Allows the instance to be used as a decorator.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self.verbose:
                    msg = f"{self.msg}{': ' if self.msg else ''}[red]{e}[/red]"
                    print_error(f"caught by [green]{func.__name__}[/green] decorator. {msg}")

        return wrapper

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        return self

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print_error(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True
