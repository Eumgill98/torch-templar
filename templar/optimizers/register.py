import torch.optim as optim

_optimizer_entrypoints = {
    'Adadelta' : optim.Adadelta,
    'Adagrad' : optim.Adagrad,
    'Adam' : optim.Adam,
    'AdamW' : optim.AdamW,
    'SparseAdam' : optim.SparseAdam,
    'Adamax' : optim.Adamax,
    'ASGD' : optim.ASGD,
    'LBFGS' : optim.LBFGS,
    'NAdam' : optim.NAdam,
    'RAdam' : optim.RAdam,
    'RMSprop' : optim.RMSprop,
    'SGD' : optim.SGD,
}

def optimizer_lists():
    """_summary_
    Return Registered criterion Key name
    """
    return list(_optimizer_entrypoints.keys())

def optimizer_entrypoint(optimizer_name):
    """_summary_

    Optimizer mapping
    """
    return _optimizer_entrypoints[optimizer_name]


def is_optimizer(optimizer_name):
    """_summary_

    Check optimizer_name
    """
    return optimizer_name in _optimizer_entrypoints


def create_optimizer(optimizer_name, **kwargs):
    """_summary_

    Args:
        optimizer_name (str): optimizer name

    Raises:
        RuntimeError: if optimizer is not in _optimizer_entrypoints 

    Returns:
        optimizer (class) : optimizer class
    """
    if is_optimizer(optimizer_name):
        create_fn = optimizer_entrypoint(optimizer_name)
        optimizer = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % optimizer_name)
    return optimizer

def optimizer_register(optimizer_name):
    """_summary_

    Custom optimizer register 
    To use decorator 

    Args:
        optimizer_name (str): Custom optimizer name 
    """
    def decorator(optimizer_class):
        _optimizer_entrypoints[optimizer_name] = optimizer_class
        return optimizer_class
    return decorator

