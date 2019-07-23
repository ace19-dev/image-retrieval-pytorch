# pylint: disable=wildcard-import, unused-wildcard-import

from .productresnet import *


__all__ = ['get_model']


def get_model(name, backbone_pretrained=False, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str
        Location for keeping the model parameters.

    Returns
    -------
    Module:
        The model.
    """
    models = {
        'product_resnet50': product_resnet50,
        'product_resnet101': product_resnet101,
        'product_cosine_softmax': product_cosine_softmax
    }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](backbone_pretrained, **kwargs)
    return net
