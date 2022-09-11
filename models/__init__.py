from models.baselines.CSRNet import CSRNet

def Model(name, *args, **kwargs):
    if name == 'CSRNet':
        return CSRNet(*args, **kwargs)
    else:
        raise NotImplementedError