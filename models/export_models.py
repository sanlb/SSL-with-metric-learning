from utils.fun_utils import export
from .CNN import CNN
from .CNN13 import CNN13

@export
def cnn13(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN13(**kwargs)
    return model


@export
def cnn(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN(**kwargs)
    return model