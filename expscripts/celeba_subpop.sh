python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/ \
    --output_dir=./results/celeba_subpop/\
    --command_launcher multi_gpu\
    --algorithms ERM IRM GroupDRO Mixup VREx     \
    --datasets WILDSCelebA  \
    --n_hparams 20       \
    --n_trials 3       \
    --steps 5000      \
    --test_envs 2 3        \
    --holdout_fraction 0 \
    --hparams '{"wilds_spu_study":1, "flip_prob":0.25}' \
    --hparams '{"wilds_spu_study":1, "flip_prob":0.1}' \
    # --hparams '{"wilds_spu_study":1}' \