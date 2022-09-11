from losses.bay_loss import Bay_Loss
from losses.ot_loss import OT_Loss
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