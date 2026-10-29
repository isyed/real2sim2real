"""Explicit memory limits for inference; defaults preserve upstream behavior."""
import os


def batch_limit(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f'{name} must be an integer') from error
    if not 1 <= value <= default:
        raise ValueError(f'{name} must be between 1 and {default}')
    return value
