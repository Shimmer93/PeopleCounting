from models.baselines.CSRNet import CSRNet, CSRResNet, CSRSwinTrans, CSRNext, CSRTwins
from models.baselines.SACANet import SACANet
from models.baselines.MAE import vgg19_trans
from models.baselines.BL import BL_VGG
from models.baselines.CCTrans import alt_gvt_large
from models.baselines.DSSINet import CRFVGG
from models.baselines.MSFANet import MSFANet
from models.baselines.MCNN import MCNN

from models.temporal.swin_transformer import VideoSwinTransformer
from models.temporal.LSTN import TemporalLSTN
from models.temporal.MLSTN import TemporalMLSTN
from models.temporal.STDNet import STDNet
from models.temporal.TAN import TAN
from models.temporal.ConvLSTM import ConvLSTM

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
    elif name == 'BL':
        return BL_VGG(*args, **kwargs)
    elif name == 'CCTrans':
        return alt_gvt_large(*args, **kwargs)
    elif name == 'DSSINet':
        return CRFVGG(*args, **kwargs)
    elif name == 'MSFANet':
        return MSFANet(*args, **kwargs)
    elif name == 'MCNN':
        return MCNN(*args, **kwargs)
    elif name == 'LSTN':
        return TemporalLSTN(*args, **kwargs)
    elif name == 'MLSTN':
        return TemporalMLSTN(*args, **kwargs)
    elif name == 'STDNet':
        return STDNet(*args, **kwargs)
    elif name == 'TAN':
        return TAN(*args, **kwargs)
    elif name == 'ConvLSTM':
        return ConvLSTM(*args, **kwargs)
    else:
        raise NotImplementedError