from .DLinear import DLinear


def select_model(name):
    model_dict = {
        'DLinear': DLinear
    }

    return model_dict[name]