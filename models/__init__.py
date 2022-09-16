from models.baselines.CSRNet import CSRNet
from models.baselines.SACANet import SACANet

def Model(name, *args, **kwargs):
    if name == 'CSRNet':
        return CSRNet(*args, **kwargs)
    if name == 'SACANet':
        return SACANet(*args, **kwargs)
    else:
        raise NotImplementedError