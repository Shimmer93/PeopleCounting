from datasets.bayesian_dataset import BayesianDataset
from datasets.binary_dataset import BinaryMapDataset
from datasets.dmap_dataset import DensityMapDataset

from datasets.bayesian_temporal_dataset import BayesianTemporalDataset

def Dataset(name, *args, **kwargs):
    if name == 'Bayesian':
        return BayesianDataset(*args, **kwargs)
    elif name == 'Binary':
        return BinaryMapDataset(*args, **kwargs)
    elif name == 'Density':
        return DensityMapDataset(*args, **kwargs)
    elif name == 'BayesianTemporal':
        return BayesianTemporalDataset(*args, **kwargs)
    else:
        raise NotImplementedError