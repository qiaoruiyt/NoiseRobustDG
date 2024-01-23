# Understanding Domain Generalization: A Noise Robustness Perspective

This repository implements the empirical studies in the paper [Understanding Domain Generalization: A Noise Robustness Perspective](https://openreview.net/forum?id=I2mIxuXA72) in ICLR 2024 by [Rui Qiao](https://qiaoruiyt.github.io) and [Bryan Kian Hsiang Low](https://www.comp.nus.edu.sg/~lowkh/research.html).

The main code is based on [Domainbed](https://github.com/facebookresearch/DomainBed) by [Gulrajani and Lopez-Paz, 2020](https://arxiv.org/abs/2007.01434). 

The study on toy example is modified from [overparam_spur_corr](https://github.com/ssagawa/overparam_spur_corr) by [Sagawa et al., 2020](https://arxiv.org/pdf/2005.04345.pdf)

### Additional datasets based on WILDS ([Koh et al., 2020](https://arxiv.org/abs/2012.07421))

* Waterbirds, Waterbirds+
* CelebA, CelebA+
* CivilComments, CivilComments+

## Main algorithms compared

* ERM
* Mixup
* GroupDRO
* IRM
* V-REx

<!-- ## Additional model selection criteria -->

## Setup

Like Domainbed, this repo can be easily setup without installing many other packages if you have already setup a Python environment with the latest PyTorch. The required packages can be installed by:
```sh
pip install -r domainbed/requirements.txt
```

## To study the label-noise robustness of DG algorithms
Choose the datasets in `domainbed/scripts/download.py`, then download the datasets (The gdown version can affect the downloading of certain datasets from Google Drive. Please consider using the recommended version in `requirements.txt`):

```sh
python3 -m domainbed.scripts.download \
       --data_dir=~/data
```

Train a model with label noise:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=~/data/\
       --algorithm ERM\
       --dataset CMNISTMod\
       --test_env 2\
       --hparams '{"flip_prob":0.1}' \
```

Launch a sweep for label noise study for standard datasets:

```sh
python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/noise_pacs/\
    --command_launcher multi_gpu\
    --algorithms IRM GroupDRO Mixup ERM VREx    \
    --datasets PACS  \
    --n_hparams 20       \
    --n_trials 3       \
    --single_test_envs         \
    --hparams '{"flip_prob":0.1}' \
```


To launch a sweep for Waterbirds, CelebA, or CivilComments datasets with 4 groups as environments, we
must specify the `--test_envs` to be `4 5`, the data processor will allocate the standard val and test splits to those envs.
The `--holdout_fraction` should be set to 0 since we are not using training domain to do model selection in this setting.
```sh
python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/noise_waterbirds/\
    --command_launcher multi_gpu\
    --algorithms ERM IRM GroupDRO Mixup VREx    \
    --datasets WILDSWaterbirds  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 4 5        \
    --holdout_fraction 0 \
    --hparams '{"flip_prob":0.1}' \
```



To launch a sweep for Waterbirds+, CelebA+, or CivilComments+ datasets with 2 newly created environments, we
must set `wilds_spu_study` to true for `--hparams` and specify the `--test_envs` to be `2 3`.
The `--holdout_fraction` should be set to 0 as well
```sh
python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/waterbirds_subpop/\
    --command_launcher multi_gpu\
    --algorithms ERM IRM GroupDRO Mixup VREx    \
    --datasets WILDSWaterbirds  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2 3        \
    --holdout_fraction 0 \
    --hparams '{"wilds_spu_study":1, "flip_prob":0.1}' \
```

### NOTE
`hparams, steps`, and other configurations affect the seed of the jobs hashed by Domainbed. The scripts provided above are minimal and may produce results with slight variations. Nonetheless, the conclusions drawn from the experiments should not be very different from the main paper. We have attached a few bash scripts used for our main experiments for easier reproducibility in `expscripts/`. 

To track the accuracy of the noisy samples in the training set, please specify `--hparams '{"wilds_spu_study":1, "flip_prob":0.1, "study_noise":1}'`

## Synthetic Toy Example

To run a simulated experiment with label noise:
```sh
cd toy_example 
python run_toy_example.py  -o results.csv -N 3000 -n 1000\
    --toy_example_name complete --p_correlation 0.99\
    --mean_causal 1 --var_causal 0.25\
    --mean_spurious 1 --var_spurious 0.25\
    --mean_noise 0 --var_noise 1 \
    --model_type logistic --error_type zero_one\
    --Lambda 1e-04 -q --d_causal 5 --d_spurious 5 \
    --label_noise 0.1
```