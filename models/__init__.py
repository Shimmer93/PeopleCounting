from models.baselines.CSRNet import CSRNet, CSRResNet, CSRSwinTrans, CSRNext, CSRTwins
from models.baselines.SACANet import SACANet
from models.baselines.MAN import vgg19_trans
from models.baselines.BL import BL_VGG
from models.baselines.CCTrans import alt_gvt_large
from models.baselines.DSSINet import CRFVGG
from models.baselines.MSFANet import MSFANet
from models.baselines.MCNN import MCNN
from models.baselines.SASNet import SASNet
from models.baselines.SASNet2 import SASNet2

from models.temporal.swin_transformer import VideoSwinTransformer
from models.temporal.LSTN import TemporalLSTN
from models.temporal.MLSTN import TemporalMLSTN
from models.temporal.STDNet import STDNet
from models.temporal.TAN import TAN
from models.temporal.ConvLSTM import ConvLSTM

from models.VCFormer.VCFormer import VCFormer
from models.LTCrowd.LTCrowd import LTCrowd
from models.baselines.CSRNet_try import TemporalCSRNet, CSRInvoNet

from models.diffusion.DiffusionCounter import DiffusionCounter
from models.ScaleCount.ScaleCount2 import CSRScaleNet, SCNet2, SCNet3
from models.ScaleCount.ScaleDensityCounter import SDCNet

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
    elif name == 'MAN':
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
    elif name == 'VCFormer':
        return VCFormer(*args, **kwargs)
    elif name == 'LTCrowd':
        return LTCrowd(*args, **kwargs)
    elif name == 'TemporalCSRNet':
        return TemporalCSRNet(*args, **kwargs)
    elif name == 'CSRInvoNet':
        return CSRInvoNet(*args, **kwargs)
    elif name == 'DiffusionCounter':
        return DiffusionCounter(*args, **kwargs)
    elif name == 'ScaleCount2':
        return SCNet2(*args, **kwargs)
    elif name == 'ScaleCount3':
        return SCNet3(*args, **kwargs)
    elif name == 'SDCNet':
        return SDCNet(*args, **kwargs)
    elif name == 'SASNet':
        return SASNet(*args, **kwargs)
    elif name == 'SASNet2':
        return SASNet2(*args, **kwargs)
    else:
        raise NotImplementedError