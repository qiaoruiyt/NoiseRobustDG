# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "${:.1f}\\pm_{{{:.1f}}}$".format(mean, err)
        # return mean, err, "{:.1f}".format(mean)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        # print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        # col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
        col_labels = ["\\textbf{" 
            + str(col_label).replace("%", "\\%").replace('&', ' ').replace('WILDS', '').replace('_', '')
            + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        # print("\\end{tabular}}")
        print("\\end{tabular}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    if 'hparams tracker' in selection_method.name.lower():
        grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_hparam": selection_method.sweep_hparam(group["records"]) }
    ).filter(lambda g: g["sweep_hparam"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset.replace('&', ' $\eta$=')))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                ####### old 
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)

                ####### new
            if None in means:
                table[i][-1] = "X"
            else:
                # table[i][-1] = "{:.1f}".format(sum(means) / len(means))
                valid_means = [m for m in means if m > 0]
                if len(valid_means) == 0:
                    table[i][-1] = "X"
                else:
                    table[i][-1] = "{:.1f}".format(sum(valid_means) / len(valid_means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")

        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            valid_means = [m for m in means if m > 0]
            if len(valid_means) == 0:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(valid_means) / len(valid_means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)


def prepare_plot_results(records, selection_method, latex=False, input_dir='default'):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    # dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    results = {'algorithms':{}}

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        errs = []
        accs = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            if mean is None or mean < 0:
                print("WARNING: None in means or mean < 0, \
                    for dataset {}, algorithm {}".format(dataset, algorithm))
            means.append(mean)
            errs.append(err)
            accs.append(list(trial_averages))
        algoritm_results = {'means': means, 'errs': errs, 'accs': accs}
        results['algorithms'][algorithm] = algoritm_results

    results['datasets'] = [d for d in dataset_names]
    results['selection_method'] = selection_method.__name__
    # save results dict with json
    result_file_name = input_dir.strip('/').split('/')[-1] + '__' + selection_method.__name__ + '.json'
    result_dir = 'plot_data'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, result_file_name), 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--noise", action="store_true", help="track noise memorization")
    parser.add_argument("--hard", action="store_true", help="track performance on hard/non-spuriously correlated samples")
    parser.add_argument("--cheat", action="store_true", help="select the best model according to the test set")
    parser.add_argument("--val_env", default=None, type=int, help="select the best model according to the validation set")
    parser.add_argument("--hparam", action="store_true", help="print the best hyperparameter setting corresponding to OracleSelectionMethod")
    parser.add_argument("--es", action="store_true", help="use early stopping")
    parser.add_argument("--no_avg", action="store_true", help="do not show average test acc")
    parser.add_argument("--save", action="store_true", help="save data as json for plotting the results")
    # parser.add_argument("--val", choices=['avg', 'worst', None], default=None, help="use validation set to select model")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = []

    if args.noise:
        if not args.es:
            SELECTION_METHODS.append(model_selection.OracleNoiseTracker)
        else:
            SELECTION_METHODS.append(model_selection.OracleESNoiseTracker)

    if args.hard:
        SELECTION_METHODS.append(model_selection.OracleHardTracker)

    # Test how well the model can be on the test set. Do not use for practice.
    if args.cheat: 
        SELECTION_METHODS.append(model_selection.CheaterSelectionMethod)

    # Track the best hparam configs
    if args.val_env is None and args.hparam:
        SELECTION_METHODS.append(model_selection.OracleHparamTracker)

    # Model selection methods for Waterbirds, CelebA, CivilComments etc. that requires specific validation set.
    if args.val_env is not None:
        SELECTION_METHODS = []
        if not args.no_avg:
            SELECTION_METHODS.append(model_selection.ValAvgTestAvg)
        if not args.es:
            if not args.no_avg:
                SELECTION_METHODS.append(model_selection.ValWGTestAvgNoES)
            SELECTION_METHODS.append(model_selection.ValWGTestWGNoES)
        else:
            if not args.no_avg:
                SELECTION_METHODS.append(model_selection.ValWGTestAvg)
            SELECTION_METHODS.append(model_selection.ValWGTestWG)
        if args.noise:
            SELECTION_METHODS.append(model_selection.ValAvgNoiseTracker)
        if args.hparam:
            SELECTION_METHODS.append(model_selection.ValWGHparamTracker)

    if len(SELECTION_METHODS) == 0:
        SELECTION_METHODS = [
            model_selection.IIDAccuracySelectionMethod,
        ]
        if not args.es:
            SELECTION_METHODS.append(model_selection.OracleSelectionMethod)
        else:
            SELECTION_METHODS.append(model_selection.OracleESSelectionMethod)

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        if args.save:
            prepare_plot_results(records, selection_method, args.latex, args.input_dir)
        else:
            print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
