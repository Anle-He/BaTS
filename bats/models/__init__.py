from .CycleNet import CycleNet
from .DLinear import DLinear
from .iTransformer import iTransformer
from .SMamba import SMamba

_MODEL_REGISTRY = {
    'CycleNet': CycleNet,
    'DLinear': DLinear,
    'iTransformer': iTransformer,
    'SMamba': SMamba,
}


def select_model(name: str) -> type:
    if name not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    return _MODEL_REGISTRY[name]
