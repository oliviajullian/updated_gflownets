import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx

from collections import namedtuple
from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian
from pathlib import Path
import argparse
import json

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

    return train, graph


def get_data(args,aux_dir):
    if args.artifact == None: #create the dataset
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
        )
        # Save data & results
        with open(aux_dir.out_dir/ 'arguments.json', 'w') as f:
            json.dump(vars(args), f, default=str)
        data.to_csv(aux_dir.out_dir/ 'train_data.csv')
        with open(aux_dir.out_dir/ 'DAG.npy', 'wb') as f:
            pickle.dump(nx.to_numpy_array(graph, weight=None).T, f)

        interventions = pd.DataFrame(False, index=data.index, columns=data.columns)
        dataset = Dataset(data=data, interventions=interventions.astype(np.float32))
        return dataset, graph

