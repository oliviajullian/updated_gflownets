import numpy as np
import pandas as pd
import networkx as nx

from numpy.random import default_rng
from pgmpy.models import LinearGaussianBayesianNetwork, BayesianNetwork


def sample_from_linear_gaussian(model, num_samples, rng=default_rng()):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if not isinstance(model, LinearGaussianBayesianNetwork):
        raise ValueError('The model must be an instance '
                         'of LinearGaussianBayesianNetwork')

    samples = pd.DataFrame(columns=list(model.nodes()))
    for node in nx.topological_sort(model):
        cpd = model.get_cpds(node)

        if cpd.evidence:
            values = np.vstack([samples[parent] for parent in cpd.evidence])
            mean = cpd.mean[0] + np.dot(cpd.mean[1:], values)
            samples[node] = rng.normal(mean, cpd.variance)
        else:
            samples[node] = rng.normal(cpd.mean[0], cpd.variance, size=(num_samples,))

    return samples
