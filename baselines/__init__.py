from .CycleNet import CycleNet


def select_model(name):
    model_dict = {
        'CycleNet': CycleNet
    }

    return model_dict[name]