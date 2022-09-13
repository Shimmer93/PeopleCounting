from datasets.bayesian_dataset import BayesianDataset
from datasets.binary_dataset import BinaryMapDataset

def Dataset(name, *args, **kwargs):
    if name == 'Bayesian':
        return BayesianDataset(*args, **kwargs)
    elif name == 'Binary':
        return BinaryMapDataset(*args, **kwargs)
    else:
        raise NotImplementedError