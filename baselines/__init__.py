from .CycleNet import CycleNet


def select_model(name):
    model_dict = {
        'SDMamba': CycleNet
    }

    return model_dict[name]