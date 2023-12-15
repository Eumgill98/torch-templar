import torch.nn as nn

_criterion_entrypoints = {
    "L1Loss" : nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "CTCLoss": nn.CTCLoss,
    "NLLLoss": nn.NLLLoss,
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SoftMarginLoss": nn.SoftMarginLoss,
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
}

def criterion_lists():
    """_summary_
    Return Registered criterion Key name
    """
    return list(_criterion_entrypoints.keys())

def criterion_entrypoint(criterion_name):
    """_summary_

    Criterion mapping
    """
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    """_summary_

    Check criterion_name
    """
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    """_summary_

    Args:
        criterion_name (str): loss name

    Raises:
        RuntimeError: if criterion is not in _criterion_entrypoints 

    Returns:
        criterion (class) : loss class
    """
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

def register_criterion(criterion_name):
    """_summary_

    Custom Loss function register 
    To use decorator 

    Args:
        criterion_name (str): Custom Loss Function name 
    """
    def decorator(criterion_class):
        _criterion_entrypoints[criterion_name] = criterion_class
        return criterion_class
    return decorator

