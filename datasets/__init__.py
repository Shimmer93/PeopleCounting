from datasets.bayesian_dataset import BayesianDataset

def Dataset(name, *args, **kwargs):
    if name == 'Bayesian':
        return BayesianDataset(*args, **kwargs)
    else:
        raise NotImplementedError