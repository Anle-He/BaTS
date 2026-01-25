from .DLinear import DLinear


def select_model(name: str) -> type:
    model_dict = {'DLinear': DLinear}

    return model_dict[name]
