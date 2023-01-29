from enum import Enum


class FrameMode(Enum):
    r"""
    Selects if instrument operating in frame-skipping mode
    """
    not_skip = 0
    skip = 1
