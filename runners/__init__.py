from .STFRunner import STFRunner
from .LTSFRunner import LTSFRunner

__all__ = ['BaseRunner', 'STFRunner', 'LTSFRunner']


def select_runner(runner:str):

    if runner == 'STFRunner':
        return STFRunner
    elif runner == 'LTSFRunner':
        return LTSFRunner
    else:
        raise NotImplementedError()