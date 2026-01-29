from .DLinear import DLinear
from .iTransformer import iTransformer
from .SMamba import SMamba


def select_model(name: str) -> type:
    model_dict = {'DLinear': DLinear, 'iTransformer': iTransformer, 'SMamba': SMamba}

    return model_dict[name]
