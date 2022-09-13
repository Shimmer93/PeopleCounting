from datasets.bayesian_dataset import BayesianDataset
from datasets.binary_dataset import BinaryMapDataset
from datasets.dmap_dataset import DensityMapDataset

def Dataset(name, *args, **kwargs):
    if name == 'Bayesian':
        return BayesianDataset(*args, **kwargs)
    elif name == 'Binary':
        return BinaryMapDataset(*args, **kwargs)
    elif name == 'Density':
        return DensityMapDataset(*args, **kwargs)
    else:
        raise NotImplementedError