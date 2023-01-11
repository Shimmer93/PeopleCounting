from datasets.bayesian_dataset import BayesianDataset
from datasets.binary_dataset import BinaryMapDataset
from datasets.dmap_dataset import DensityMapDataset
from datasets.smap_dataset import ScaleMapDataset
from datasets.smap_dataset2 import ScaleMapDataset2
from datasets.scale_bayesian_dataset import ScaleBayesianDataset
from datasets.scale_select_dataset import ScaleSelectDataset

from datasets.bayesian_temporal_dataset import BayesianTemporalDataset
from datasets.dmap_temporal_dataset import DensityTemporalDataset

def Dataset(name, *args, **kwargs):
    if name == 'Bayesian':
        return BayesianDataset(*args, **kwargs)
    elif name == 'Binary':
        return BinaryMapDataset(*args, **kwargs)
    elif name == 'Density':
        return DensityMapDataset(*args, **kwargs)
    elif name == 'BayesianTemporal':
        return BayesianTemporalDataset(*args, **kwargs)
    elif name == 'DensityTemporal':
        return DensityTemporalDataset(*args, **kwargs)
    elif name == 'Scale':
        return ScaleMapDataset(*args, **kwargs)
    elif name == 'Scale2':
        return ScaleMapDataset2(*args, **kwargs)
    elif name == 'ScaleBayesian':
        return ScaleBayesianDataset(*args, **kwargs)
    elif name == 'ScaleSelect':
        return ScaleSelectDataset(*args, **kwargs)
    else:
        raise NotImplementedError