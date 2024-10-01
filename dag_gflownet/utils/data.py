import pandas as pd
import numpy as np
import pickle
import os

from collections import namedtuple

Dataset = namedtuple('Dataset', ['data', 'interventions'])


def load_artifact_dataset(data, artifact_dir, prefix):
    
    mapping_filename = f"{artifact_dir}/intervention_mapping.csv"
    filename = f"{artifact_dir}/{prefix}_interventions.csv"

    if os.path.exists(filename) and os.path.exists(mapping_filename):
        mapping = pd.read_csv(mapping_filename, index_col=0, header=0)
        perturbations = pd.read_csv(filename, index_col=0, header=0)

        interventions = perturbations.dot(mapping.reindex(index=perturbations.columns))
        interventions = interventions.reindex(columns=data.columns)
    else:
        interventions = pd.DataFrame(False, index=data.index, columns=data.columns)

    return Dataset(data=data, interventions=interventions.astype(np.float32))


def load_artifact_continuous(artifact_dir):
    filename = f"{artifact_dir}/graph.pkl"
    if os.path.exists(filename):
        with open(artifact_dir / 'graph.pkl', 'rb') as f:
            graph = pickle.load(f)
    else:
        graph= None

    train_data = pd.read_csv(f"{artifact_dir}/train_data.csv")
    train = load_artifact_dataset(train_data, artifact_dir, 'train')

    filename = f"{artifact_dir}/valid_data.csv"
    if os.path.exists(filename):
        valid_data = pd.read_csv(artifact_dir / 'valid_data.csv', index_col=0, header=0)
        valid = load_artifact_dataset(valid_data, artifact_dir, 'valid')
    else:
        valid = None

    return train, valid, graph
