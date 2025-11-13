import os
import sys
from contextlib import contextmanager


@contextmanager
def silenced_output():
    """
    A context manager to suppress all stdout and stderr,
    including from C-level libraries.
    """
    # Open a null file descriptor
    devnull_fd = os.open(os.devnull, os.O_RDWR)

    # Save the original stdout and stderr file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Duplicate the original file descriptors to save them
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    try:
        # Redirect Python's stdout/stderr file descriptors
        # to the null device
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, original_stdout_fd)
        os.dup2(devnull_fd, original_stderr_fd)

        # Yield control back to the 'with' block
        yield
    finally:
        # Restore the original stdout/stderr from the saved FDs
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Close the temporary FDs
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
