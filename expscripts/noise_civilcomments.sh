python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/ \
    --output_dir=./results/noise_civilcomments/\
    --command_launcher multi_gpu\
    --algorithms ERM IRM GroupDRO VREx    \
    --datasets WILDSCivilComments  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 4 5        \
    --holdout_fraction 0 \
    # --hparams '{"flip_prob":0.1, "study_noise":1}' \
    # --hparams '{"flip_prob":0.25, "study_noise":1}' \
