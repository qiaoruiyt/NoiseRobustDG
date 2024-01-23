"""
Example usage:
python -u -m domainbed.scripts.plot_results \
    --input_dir ./results/cmnist
"""

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings


import matplotlib.pyplot as plt


def plot_data(data, dataset, train_data=None, val_data=None):
    # data should be in format # {'algorithm':[seed, group, trajectory]} or {'algorithm':[seed, trajectory]}
    # colors=plt.cm.get_cmap('tab10').colors
    plot_dir = os.path.join('figures', dataset)
    os.makedirs(plot_dir, exist_ok=True)
    for alg in data.keys():
        num_plots = data[alg].shape[0]
        for i in range(data[alg].shape[0]):
            plt.figure(figsize=(10, 10))
            trial_i = data[alg][i]
            if trial_i.ndim == 2:
                for j in range(trial_i.shape[0]):
                    trajectory = trial_i[j]
                    plt.plot(trajectory, label=alg+f'_test_group{j}')
            else:
                plt.plot(trial_i, label=alg+'_test')
            
            if val_data is not None:
                val_trial_i = val_data[alg][i]
                if val_trial_i.ndim == 2:
                    for j in range(val_trial_i.shape[0]):
                        trajectory = val_trial_i[j]
                        plt.plot(trajectory, label=alg+f'_val_group{j}')
                else:
                    plt.plot(val_trial_i, label=alg+'_val')

            if train_data is not None:
                train_trial_i = train_data[alg][i]
                if train_trial_i.ndim == 2:
                    for j in range(train_trial_i.shape[0]):
                        trajectory = train_trial_i[j]
                        plt.plot(trajectory, label=alg+f'_train_group{j}')
                else:
                    plt.plot(train_trial_i, label=alg+'_train')

            # break # only plot the first seed 
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            plt.title('Trajectories of different groups')
            plt.savefig(os.path.join(plot_dir, f'{alg}_trajectories_seed{i}.png'))



def plot_result_trajectories(records, selection_method, latex, test_env=None, track_train=False):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    if 'hparams tracker' in selection_method.name.lower():
        print("Tracking hparams")
        grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "val_trajectories": selection_method.sweep_trajectory(group["records"])[0],
        "test_trajectories": selection_method.sweep_trajectory(group["records"])[1],
        "train_trajectories": selection_method.sweep_trajectory(group["records"])[2]}
        ).filter(lambda g: g["test_trajectories"] is not None)

    print("Total groups:", len(grouped_records))

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    # dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    test_trajectories = {} # {'algorithm':[seed, group, trajectory]}
    val_trajectories = {}
    train_trajectories = {}

    for dataset in dataset_names:
        if test_env is None:
            test_envs = range(datasets.num_environments(dataset))
        else:
            test_envs = [test_env]

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_trajectory_test = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("test_trajectories"))

                trial_trajectory_val = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("val_trajectories"))

                trial_trajectory_train = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("train_trajectories"))
                    
                if len(trial_trajectory_test) != 0:
                    print(dataset, algorithm, test_env)
                    trial_trajectory_test = np.array(trial_trajectory_test)
                    test_trajectories[algorithm] = trial_trajectory_test
                    val_trajectories[algorithm] = np.array(trial_trajectory_val)
                    train_trajectories[algorithm] = np.array(trial_trajectory_train)
                    # print(trial_hparam)
                    break

                # base on the selected result, extract the trajectories
        if track_train:
            plot_data(test_trajectories, dataset, train_trajectories)
        else:
            plot_data(test_trajectories, dataset)
        # plot_data(test_trajectories, dataset, val_trajectories)
        

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--val_env", default=None, type=int, help="select the best model according to the validation set")
    parser.add_argument("--hparam", action="store_true", help="print the best hyperparameter setting corresponding to OracleSelectionMethod")
    parser.add_argument("--es", action="store_true", help="use early stopping")
    parser.add_argument("--test_env", default=None, type=str, help="select the test environments")
    parser.add_argument("--train", action="store_true", help="track the training accuracy")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    SELECTION_METHODS = []
    if args.val_env is None:
        selection_method = model_selection.OracleHparamTracker
    else:
        selection_method = model_selection.ValWGHparamTracker

    if args.es:
        try:
            selection_method.es = True
        except:
            print("Early stopping not supported for this selection method, using last step instead")

    plot_result_trajectories(records, selection_method, args.latex, args.test_env, track_train=args.train)
