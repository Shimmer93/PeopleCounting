from models.baselines.CSRNet import CSRNet, CSRResNet, CSRSwinTrans, CSRNext, CSRTwins
from models.baselines.SACANet import SACANet
from models.baselines.MAE import vgg19_trans
from models.baselines.HRFormer import HRFormer
from models.baselines.HigherHRNet import HigherHRNet
from models.baselines.BL import BL_VGG
from models.baselines.CCTrans import alt_gvt_large
from models.baselines.DSSINet import CRFVGG
from models.baselines.MSFANet import MSFANet

from models.temporal.swin_transformer import VideoSwinTransformer

def Model(name, *args, **kwargs):
    if name == 'CSRNet':
        return CSRNet(*args, **kwargs)
    elif name == 'CSRResNet':
        return CSRResNet(*args, **kwargs)
    elif name == 'CSRSwinTrans':
        return CSRSwinTrans(*args, **kwargs)
    elif name == 'CSRNext':
        return CSRNext(*args, **kwargs)
    elif name == 'CSRTwins':
        return CSRTwins(*args, **kwargs)
    elif name == 'SACANet':
        return SACANet(*args, **kwargs)
    elif name == 'MAE':
        return vgg19_trans(*args, **kwargs)
    elif name == 'VideoSwin':
        return VideoSwinTransformer(*args, **kwargs)
    elif name == 'HRFormer':
        return HRFormer(*args, **kwargs)
    elif name == 'HigherHRNet':
        return HigherHRNet(*args, **kwargs)
    elif name == 'BL':
        return BL_VGG(*args, **kwargs)
    elif name == 'CCTrans':
        return alt_gvt_large(*args, **kwargs)
    elif name == 'DSSINet':
        return CRFVGG(*args, **kwargs)
    elif name == 'MSFANet':
        return MSFANet(*args, **kwargs)
    else:
        raise NotImplementedError