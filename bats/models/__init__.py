from .CAM import CAM
from .CycleNet import CycleNet
from .DLinear import DLinear
from .DSTMamba import DSTMamba
from .iTransformer import iTransformer
from .SMamba import SMamba
from .STID import STID

_MODEL_REGISTRY = {
    'CAM': CAM,
    'CycleNet': CycleNet,
    'DLinear': DLinear,
    'DSTMamba': DSTMamba,
    'iTransformer': iTransformer,
    'SMamba': SMamba,
    'STID': STID,
}


def select_model(name: str) -> type:
    if name not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    return _MODEL_REGISTRY[name]
