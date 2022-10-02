from models.baselines.CSRNet import CSRNet
from models.baselines.SACANet import SACANet
from models.baselines.MAE import vgg19_trans

from models.temporal.swin_transformer import VideoSwinTransformer

def Model(name, *args, **kwargs):
    if name == 'CSRNet':
        return CSRNet(*args, **kwargs)
    elif name == 'SACANet':
        return SACANet(*args, **kwargs)
    elif name == 'MAE':
        return vgg19_trans(*args, **kwargs)
    elif name == 'VideoSwin':
        return VideoSwinTransformer(*args, **kwargs)
    else:
        raise NotImplementedError