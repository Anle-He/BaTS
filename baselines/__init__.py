from .SDMamba import SDMamba


def select_model(name):
    model_dict = {
        'SDMamba': SDMamba
    }

    return model_dict[name]