python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/ \
    --output_dir=./results/civilcomments_subpop/\
    --command_launcher multi_gpu\
    --algorithms ERM IRM GroupDRO VREx    \
    --datasets WILDSCivilComments  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2 3        \
    --holdout_fraction 0 \
    --hparams '{"wilds_spu_study":1}' \
#     --hparams '{"wilds_spu_study":1, "flip_prob":0.1}' \
    # --hparams '{"wilds_spu_study":1, "flip_prob":0.25}' \
