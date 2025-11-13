from rich import print as rprint

from .constants import IS_GLOBAL_ZERO

__all__ = ["print_error", "print_info", "print_warning", "print", "print_warning_once"]

printed_warnings = set()


def print_error(text="", only_zero_rank=False):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[red bold]ERROR: [/red bold]{text}")


def print_warning(text="", only_zero_rank=False):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[yellow bold]WARNING: [/yellow bold]{text}")


def print_warning_once(text="", only_zero_rank=False):
    global printed_warnings
    if text in printed_warnings:
        return
    printed_warnings.add(text)
    print_warning(text, only_zero_rank)


def print_info(text="", only_zero_rank=True):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[blue bold]INFO: [/blue bold]{text}")


def print(text="", only_zero_rank=True):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(text)
