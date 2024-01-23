python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/ \
    --output_dir=./results/waterbirds_subpop/\
    --command_launcher multi_gpu\
    --algorithms ERM IRM GroupDRO Mixup VREx    \
    --datasets WILDSWaterbirds  \
    --n_hparams 20       \
    --n_trials 3       \
    --steps 5000      \
    --test_envs 2 3        \
    --holdout_fraction 0 \
    --hparams '{"wilds_spu_study":1}' \
    # --hparams '{"wilds_spu_study":1, "flip_prob":0.25, "study_noise":1}' \
    # --hparams '{"wilds_spu_study":1, "flip_prob":0.1, "study_noise":1}' \
    # --skip_confirmation