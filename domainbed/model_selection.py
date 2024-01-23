# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np
import json

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def trial_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.trial_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    # new
    # @classmethod
    # def sweep_acc(self, records):
    #     """
    #     Given all records from a single (dataset, algorithm, test env) pair,
    #     return the mean test acc of the k runs with the top val accs.
    #     """
    #     _trial_accs = self.trial_accs(records)
    #     if len(_trial_accs):
    #         # print(len(_trial_accs))
    #         trial_accs = [_trial_accs[i][0]['test_acc'] for i in range(len(_trial_accs))]
    #         return trial_accs
    #     else:
    #         return None

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None
        
    @classmethod
    def sweep_hparam(self, records):
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['record_dir']
        else:
            return None

    @classmethod
    def sweep_trajectory(self, records):
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['val_trajectories'], \
                _hparams_accs[0][0]['test_trajectories'], \
                _hparams_accs[0][0]['train_trajectories']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle) w/o es"

    @classmethod
    def run_acc(self, run_records):
        # NOTE: only keeping the records with single test env (default domainbed implementation)
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        #NOTE: Only choosing the first test env, so it's not applicable to multiple test envs
        # The only time of choosing at most 2 test envs is for leave-one-out model selection. 
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class OracleESSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all best
    checkpoints, i.e. with (simulated) early stopping."""
    name = "test-domain validation set (oracle) w/ es"

    @classmethod
    def run_acc(self, run_records):
        # NOTE: only keeping the records with single test env (default domainbed implementation)
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        #NOTE: Only choosing the first test env, so it's not applicable to multiple test envs
        # The only time of choosing at most 2 test envs is for leave-one-out model selection. 
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r[test_out_acc_key])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }


class OracleESNoiseTracker(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all best
    checkpoints, i.e. with (simulated) early stopping.
    Currently, this method only tracks the noise of the first train_env, need to change it to track all train_envs
    """
    name = "Noisy Samples Tracker w/ ES"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        #NOTE: Only choosing the first test env, so it's not applicable to multiple test envs
        # The only time of choosing at most 2 test envs is for leave-one-out model selection. 
        test_env = run_records[0]['args']['test_envs'][0]

        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r[test_out_acc_key])[-1]
        
        #NOTE: This average the noisy set acc across all train_envs regardless of the number of noisy samples in each env
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        try:
            n = sum([1 for key in chosen_record.keys() if 'noisy' in key])
            group_accs = []
            for i in range(n):
                train_noise_acc_key = 'env{}_noisy_acc'.format(i)
                train_noise_acc = chosen_record[train_noise_acc_key]
                group_accs.append(train_noise_acc)
            noisy_set_acc = sum(group_accs)/n
        except:
            noisy_set_acc = -1

        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': noisy_set_acc
        }

class OracleNoiseTracker(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping.
    Currently, this method only tracks the noise of the first train_env, need to change it to track all train_envs
    """
    name = "Noisy Samples Tracker w/o ES"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        #NOTE: Only choosing the first test env, so it's not applicable to multiple test envs
        # The only time of choosing at most 2 test envs is for leave-one-out model selection. 
        test_env = run_records[0]['args']['test_envs'][0]

        #NOTE: This average the noisy set acc across all train_envs regardless of the number of noisy samples in each env
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        try:
            n = sum([1 for key in chosen_record.keys() if 'noisy' in key])
            group_accs = []
            for i in range(n):
                train_noise_acc_key = 'env{}_noisy_acc'.format(i)
                train_noise_acc = chosen_record[train_noise_acc_key]
                group_accs.append(train_noise_acc)
            noisy_set_acc = sum(group_accs)/n
        except:
            noisy_set_acc = -1

        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': noisy_set_acc
        }
        

class OracleHardTracker(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "Hard Samples Tracker"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        #NOTE: Only choosing the first test env, so it's not applicable to multiple test envs
        # The only time of choosing at most 2 test envs is for leave-one-out model selection. 
        test_env = run_records[0]['args']['test_envs'][0]
        train_env = 0
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        train_hard_set_acc_key = 'env{}_hard_acc'.format(train_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        try:
            test_acc = chosen_record[train_hard_set_acc_key]
        except:
            test_acc = -1

        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': test_acc
        }


class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None


class OracleHparamTracker(SelectionMethod):
    """This backtracks the best hparams for each test env
    """
    name = "Hparams Tracker"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        #NOTE: Only choosing the first test env, so it's not applicable to multiple test envs
        # The only time of choosing at most 2 test envs is for leave-one-out model selection. 
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        hparams_key = 'hparams'
        record_dir_key = 'args'
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]

        val_trajectories = []
        test_trajectories = []
        train_trajectories = [] 
            
        for j in range(len(run_records)):
            chosen_record = run_records[j]
            test_group_acc = chosen_record[test_out_acc_key]
            test_trajectories.append(test_group_acc)

            val_group_acc = chosen_record[test_in_acc_key]
            val_trajectories.append(val_group_acc)

        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key],
            'hparams': chosen_record[hparams_key],
            'record_dir': chosen_record[record_dir_key],
            'test_env': run_records[0]['args']['test_envs'],
            'val_trajectories': val_trajectories,
            'test_trajectories': test_trajectories,
            'train_trajectories': train_trajectories,
        }


    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None


# Designed for Waterbirds, CelebA and WILDS Datasets Only
# We do not hold out any splits for train set. 
# The second last env is for val, and the last env is for test

class ValAvgTestAvg(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "Avg Acc with Avg Val Selection (w Early Stopping)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # NOTE: by construction, the first test_env is val_set, the rest are test_set
        val_env = run_records[0]['args']['test_envs'][0] 
        val_in_acc_key = 'env{}_in_acc'.format(val_env) # val acc
        chosen_record = run_records.sorted(lambda r: r[val_in_acc_key])[-1]

        # NOTE: So far we assume that there is only one test env named with env{}_in_acc (for the average case)
        test_env = run_records[0]['args']['test_envs'][1]
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test acc
        return {
            'val_acc':  chosen_record[val_in_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }
    
class ValWGTestWG(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "WG Acc with WG Val Selection (w Early Stopping))"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # count the number of keys that start with "val" in the chosen_record
        n = sum([1 for key in run_records[0].keys() if key.startswith('val')])
        worst_group_accs = []
        for i in range(len(run_records)):
            chosen_record = run_records[i]
            group_accs = []
            for j in range(n):
                val_group_acc_key = 'val{}_acc'.format(j)
                val_group_acc = chosen_record[val_group_acc_key]
                group_accs.append(val_group_acc)
            worst_acc = min(group_accs)
            worst_group_accs.append(worst_acc)
        chosen_idx = np.argmax(worst_group_accs)
        
        chosen_record = run_records[chosen_idx]
        worst_group_val_acc = worst_group_accs[chosen_idx]

        # count the number of keys that start with "test" in the chosen_record
        n = sum([1 for key in chosen_record.keys() if key.startswith('test')])
        group_accs = []

        # grouped test envs are named test{}_acc
        for i in range(n):
            test_group_acc_key = 'test{}_acc'.format(i)
            test_group_acc = chosen_record[test_group_acc_key]
            group_accs.append(test_group_acc)
        worst_test_group_acc = min(group_accs)
        return {
            'val_acc':  worst_group_val_acc,
            'test_acc': worst_test_group_acc
        }


class ValWGTestAvg(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "Avg Test Acc with WG Val Selection (w Early Stopping))"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # count the number of keys that start with "val" in the chosen_record
        n = sum([1 for key in run_records[0].keys() if key.startswith('val')])
        worst_group_accs = []
        for i in range(len(run_records)):
            chosen_record = run_records[i]
            group_accs = []
            for j in range(n):
                val_group_acc_key = 'val{}_acc'.format(j)
                val_group_acc = chosen_record[val_group_acc_key]
                group_accs.append(val_group_acc)
            worst_acc = min(group_accs)
            worst_group_accs.append(worst_acc)
        chosen_idx = np.argmax(worst_group_accs)
        
        chosen_record = run_records[chosen_idx]
        worst_group_val_acc = worst_group_accs[chosen_idx]

        # NOTE: So far we assume that there is only one test env
        test_env = run_records[0]['args']['test_envs'][1]
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test acc
        return {
            'val_acc':  worst_group_val_acc,
            'test_acc': chosen_record[test_in_acc_key]
        }


class ValAvgTestAvgNoES(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "Avg Acc with Avg Val Selection (w/o Early Stopping)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # NOTE: by construction, the first test_env is val_set, the rest are test_set
        val_env = run_records[0]['args']['test_envs'][0] 
        val_in_acc_key = 'env{}_in_acc'.format(val_env) # val acc
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]

        # NOTE: So far we assume that there is only one test env
        test_env = run_records[0]['args']['test_envs'][1]
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test acc
        return {
            'val_acc':  chosen_record[val_in_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }
    
class ValWGTestWGNoES(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "WG Acc with WG Val Selection (w/o Early Stopping))"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        chosen_record = run_records.sorted(lambda r: r['step'])[-1]

        # count the number of keys that start with "val" in the chosen_record
        n = sum([1 for key in chosen_record.keys() if key.startswith('val')])
        group_accs = []
        for j in range(n):
            val_group_acc_key = 'val{}_acc'.format(j)
            val_group_acc = chosen_record[val_group_acc_key]
            group_accs.append(val_group_acc)
        worst_group_val_acc = min(group_accs)

        # count the number of keys that start with "test" in the chosen_record
        n = sum([1 for key in chosen_record.keys() if key.startswith('test')])
        group_accs = []
        for i in range(n):
            test_group_acc_key = 'test{}_acc'.format(i)
            test_group_acc = chosen_record[test_group_acc_key]
            group_accs.append(test_group_acc)
        worst_test_group_acc = min(group_accs)
        return {
            'val_acc':  worst_group_val_acc,
            'test_acc': worst_test_group_acc
        }

class ValWGTestAvgNoES(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "Avg Test Acc with WG Val Selection (w/o Early Stopping))"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        chosen_record = run_records.sorted(lambda r: r['step'])[-1]

        # count the number of keys that start with "val" in the chosen_record
        n = sum([1 for key in chosen_record.keys() if key.startswith('val')])
        group_accs = []
        for j in range(n):
            val_group_acc_key = 'val{}_acc'.format(j)
            val_group_acc = chosen_record[val_group_acc_key]
            group_accs.append(val_group_acc)
        worst_group_val_acc = min(group_accs)

        # NOTE: So far we assume that there is only one test env
        test_env = run_records[0]['args']['test_envs'][1]
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test acc
        return {
            'val_acc':  worst_group_val_acc,
            'test_acc': chosen_record[test_in_acc_key]
        }


class ValAvgNoiseTracker(SelectionMethod):
    """This follows the standard model selection method for Waterbirds, CelebA, and WILDS Datasets. 
    We train the models on the training envs and select the best model based on the validation envs.
    By design, we put the validation env as the first test_envs and the test env in the rest of the entries."""
    name = "Training Set Average Noisy Data Memorization Tracking"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # NOTE: by construction, the first test_env is val_set, the rest are test_set
        val_env = run_records[0]['args']['test_envs'][0] 
        val_in_acc_key = 'env{}_in_acc'.format(val_env) # NOTE: this is the val acc
        chosen_record = run_records.sorted(lambda r: r[val_in_acc_key])[-1]
        
        #NOTE: This average the noisy set acc across all train_envs regardless of the number of noisy samples in each env
        try:
            n = sum([1 for key in chosen_record.keys() if 'noisy' in key])
            group_accs = []
            for i in range(n):
                train_noise_acc_key = 'env{}_noisy_acc'.format(i)
                train_noise_acc = chosen_record[train_noise_acc_key]
                group_accs.append(train_noise_acc)
            noisy_set_acc = sum(group_accs)/n
        except:
            noisy_set_acc = -1

        return {
            'val_acc':  chosen_record[val_in_acc_key],
            'test_acc': noisy_set_acc
        }
    
class ValAvgHparamTracker(SelectionMethod):
    """This is a tracker that backtrack the best hparams for each test env
    """
    name = "Hparams Tracker for Best Avg Val Selection"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # NOTE: by construction, the first test_env is val_set, the rest are test_set
        val_env = run_records[0]['args']['test_envs'][0] 
        val_in_acc_key = 'env{}_in_acc'.format(val_env) # NOTE: this is the val acc
        chosen_record = run_records.sorted(lambda r: r[val_in_acc_key])[-1]
        
        test_env = run_records[0]['args']['test_envs'][1]
        test_in_acc_key = 'env{}_in_acc'.format(test_env) # test acc

        hparams_key = 'hparams'
        record_dir_key = 'args'
        return {
            'val_acc':  chosen_record[val_in_acc_key],
            'test_acc': chosen_record[test_in_acc_key],
            'hparams': chosen_record[hparams_key],
            'record_dir': chosen_record[record_dir_key],
            'test_env': run_records[0]['args']['test_envs']
        }
    
    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        # NOTE: it feels like top 1 val acc for all hparam candidates
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None



class ValWGHparamTracker(SelectionMethod):
    """This is a tracker that backtrack the best hparams for each test env
    """
    name = "Hparams Tracker for Best WG Val Selection"
    es = False

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) > 1)
        if not len(run_records):
            print("No validation environment found! Test envs: {}".format(run_records[0]['args']['test_envs']))
            return None

        # count the number of keys that start with "val" in the chosen_record
        n = sum([1 for key in run_records[0].keys() if key.startswith('val')])
        worst_group_accs = []
        test_trajectories = []
        val_trajectories = []

        for i in range(len(run_records)):
            chosen_record = run_records[i]
            group_accs = []
            group_val_trajectory = []
            for j in range(n):
                val_group_acc_key = 'val{}_acc'.format(j)
                val_group_acc = chosen_record[val_group_acc_key]
                group_accs.append(val_group_acc)

                group_val_trajectory.append(chosen_record[val_group_acc_key])
            val_trajectories.append(group_val_trajectory)

            worst_acc = min(group_accs)
            worst_group_accs.append(worst_acc)

        chosen_idx = np.argmax(worst_group_accs)
        if not self.es:
            chosen_idx = -1
        
        chosen_record = run_records[chosen_idx]
        worst_group_val_acc = worst_group_accs[chosen_idx]

        # count the number of keys that start with "test" in the chosen_record
        n = sum([1 for key in chosen_record.keys() if key.startswith('test')])
        group_accs = []
        for i in range(n):
            test_group_acc_key = 'test{}_acc'.format(i)
            test_group_acc = chosen_record[test_group_acc_key]
            group_accs.append(test_group_acc)

            group_test_trajectory = []
            for j in range(len(run_records)):
                chosen_record = run_records[j]
                test_group_acc = chosen_record[test_group_acc_key]
                group_test_trajectory.append(test_group_acc)
            test_trajectories.append(group_test_trajectory)

        worst_test_group_acc = min(group_accs)

        ### track training trajectories
        # count the number of keys that has "in_acc" in the chosen_record
        n = sum([1 for key in chosen_record.keys() if 'in_acc' in key])
        train_trajectories = []
        if n > 0:
            for i in range(n):
                train_group_acc_key = 'env{}_in_acc'.format(i)
                train_group_acc = chosen_record[train_group_acc_key]

                group_train_trajectory = []
                for j in range(len(run_records)):
                    chosen_record = run_records[j]
                    train_group_acc = chosen_record[train_group_acc_key]
                    group_train_trajectory.append(train_group_acc)
                train_trajectories.append(group_train_trajectory)


        hparams_key = 'hparams'
        record_dir_key = 'args'
        return {
            'val_acc':  worst_group_val_acc,
            'test_acc': worst_test_group_acc,
            'hparams': chosen_record[hparams_key],
            'record_dir': chosen_record[record_dir_key]['output_dir'],
            'test_env': run_records[0]['args']['test_envs'],
            'test_trajectories': test_trajectories,
            'val_trajectories': val_trajectories,
            'train_trajectories': train_trajectories,
        }
    
    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None