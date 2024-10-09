import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import optax
import jax
import networkx as nx
import json
import argparse
from pathlib import Path
from tqdm import trange
from numpy.random import default_rng

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_model, get_model_prior
from dag_gflownet.utils.gflownet import posterior_estimate,get_most_likely_graph
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils.data import load_artifact_continuous, get_data
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics



def main(args):

    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    #for saving results
    aux_dir = argparse.Namespace(out_dir=Path(args.out_dir))
    aux_dir.out_dir.mkdir(exist_ok=True)

    # Load or create data & graph
    train, graph = get_data(args,aux_dir)

    if args.artifact==None:
        gt_graph = nx.to_numpy_array(graph, weight=None)
    else:
        gt_graph = graph.T

    # Load data & graph
    train_jnp = jax.tree_util.tree_map(jnp.asarray, train)

    # Create the environment
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        num_variables=args.num_variables,
        max_parents=args.max_parents
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables,
    )

    # Create the model
    if args.obs_scale is None:
        raise ValueError('The obs_noise is not defined in the artifact, '
            'therefore is must be set as a command argument `--obs_scale`.')
    obs_scale = args.obs_scale
    #args for the creation of the graph
    metadata = {"num_variables":args.num_variables,"num_edges_per_node":args.num_edges_per_node}

    prior_graph = get_model_prior(args.prior, metadata, args)
    model = get_model(args.model, prior_graph, train_jnp, obs_scale)

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(
        model=model,
        delta=args.delta,
        num_samples=args.params_num_samples,
        update_target_every=args.update_target_every,
        dataset_size=train_jnp.data.shape[0],
        batch_size=args.batch_size_data,
    )

    optimizer = optax.adam(args.lr)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['graph'],
        replay.dummy['mask']
    )

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),
        transition_steps=args.num_iterations // 2,
        transition_begin=args.prefill,
    ))

    # Training loop
    indices = None
    observations = env.reset()
    normalization = jnp.array(train_jnp.data.shape[0])

    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, logs = gflownet.act(params.online, key, observations, epsilon, normalization)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = gflownet.step(params, state, samples, train_jnp, normalization)

                pbar.set_postfix(loss=f"{logs['loss']:.2f}")
                # Evaluate the posterior estimate
                posterior, logs = posterior_estimate(
                    gflownet,
                    params.online,
                    env,
                    key,
                    train_jnp,
                    num_samples=args.num_samples_posterior,
                    desc='Sampling from posterior'
                )
                if iteration-args.prefill% 100:
                    results = {
                        'expected_shd': expected_shd(posterior, gt_graph),
                        'expected_edges': expected_edges(posterior),
                        **threshold_metrics(posterior, gt_graph)
                    }
                    print(f"expected_shd: {results['expected_shd']}, expected_edges: {results['expected_edges']}, roc:{results['roc_auc']} ")
                    #print the resulted maps
                    pred_1=get_most_likely_graph(posterior)
                    png_dir = aux_dir.out_dir.joinpath(f"maps/")
                    png_dir.mkdir(exist_ok=True)
                    display_matrices(pred_1, gt_graph,results,png_dir.joinpath(f"map_for_it_{iteration-args.prefill}.png"))


    """
    # Evaluate the posterior estimate
    posterior, logs = posterior_estimate(
        gflownet,
        params.online,
        env,
        key,
        train_jnp,
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior'
    )
    """
    

    results = {
        'expected_shd': expected_shd(posterior, gt_graph),
        'expected_edges': expected_edges(posterior),
        **threshold_metrics(posterior, gt_graph)
    }

    # Save model data & results
    """io.save(aux_dir.out_dir / 'model.npz', params=params.online)
    replay.save(aux_dir.out_dir / 'replay_buffer.npz')"""
    np.save(aux_dir.out_dir / 'posterior.npy', posterior)
    with open(aux_dir.out_dir / 'results.json', 'w') as f:
        json.dump(results, f, default=list)

# Function to display matrices using black and white
def display_matrices(mat1, mat2,results,dir_):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Show first matrix
    ax1.imshow(mat1, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f"pred with SHD:{results['expected_shd']}")

    # Show second matrix
    ax2.imshow(mat2, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("real")
    plt.savefig(f"{dir_}")
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    import json
    import math

    parser = ArgumentParser(description='JSP-GFN for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=1,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--max_parents', type=int, default=None,
        help='Maximum number of parents')
    environment.add_argument('--num_variables', type=int, default=6,
        help='Number of variables')
    environment.add_argument('--num_edges_per_node', type=int, default=6,
        help='Number of edges_per_node')
    environment.add_argument('--num_edges', type=int, default=6,
        help='Number of edges_per_node')

    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--artifact', type=str, default=None,
        help='Path to the artifact for input data in Wandb')
    data.add_argument('--obs_scale', type=float, default=math.sqrt(0.1),
        help='Scale of the observation noise (default: %(default)s)')
    data.add_argument('--num_samples', type=float, default=5000,
        help='Number of samples in case we create a dataset (default: %(default)s)')

    # Model
    model = parser.add_argument_group('Model')
    model.add_argument('--model', type=str, default='lingauss_diag',
        choices=['lingauss_diag', 'lingauss_full', 'mlp_gauss'],
        help='Type of model (default: %(default)s)')
    

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=10,
        help='Number of iterations (default: %(default)s)')
    optimization.add_argument('--params_num_samples', type=int, default=1,
        help='Number of samples of model parameters to compute the loss (default: %(default)s)')
    optimization.add_argument('--update_target_every', type=int, default=0,
        help='Frequency of update for the target network (0 = no target network)')
    optimization.add_argument('--batch_size_data', type=int, default=None,
        help='Batch size for the data (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=100,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')

    # Results
    results = parser.add_argument_group('Results')
    results.add_argument('--out_dir', type=str, required=True,
        help='Directory to save the results')
    

    args = parser.parse_args()

    main(args)
