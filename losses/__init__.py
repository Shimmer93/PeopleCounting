from losses.bay_loss import Bay_Loss
from losses.ot_loss import OT_Loss
from losses.dms_ssim import NORMMSSSIM
from losses.lstn_loss import LSTN_Loss, MLSTN_Loss
from losses.tan_loss import TAN_Loss
from losses.prl import PRL
from losses.lt_loss import LTLoss
from losses.sdc_loss import SDCLoss

from torch.nn import MSELoss, L1Loss

def Loss(name, *args, **kwargs):
    if name == 'MSE':
        return MSELoss(*args, **kwargs)
    elif name in ['MAE', 'L1']:
        return L1Loss(*args, **kwargs)
    elif name in ['Bay', 'Bayesian']:
        return Bay_Loss(*args, **kwargs)
    elif name == 'OT':
        return OT_Loss(*args, **kwargs)
    elif name == 'DMS_SSIM':
        return NORMMSSSIM(*args, **kwargs)
    elif name == 'LSTN':
        return LSTN_Loss(*args, **kwargs)
    elif name == 'MLSTN':
        return MLSTN_Loss(*args, **kwargs)
    elif name == 'TAN':
        return TAN_Loss(*args, **kwargs)
    elif name == 'PRL':
        return PRL(*args, **kwargs)
    elif name == 'LT':
        return LTLoss(*args, **kwargs)
    elif name == 'SDC':
        return SDCLoss(*args, **kwargs)