python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/noise_pacs/\
    --command_launcher multi_gpu\
    --algorithms IRM GroupDRO Mixup ERM VREx     \
    --datasets PACS  \
    --n_hparams 20       \
    --n_trials 3       \
    --steps 5000      \
    --single_test_envs         \
    --hparams '{"flip_prob":0.25, "study_noise":1}'
    # --hparams '{"flip_prob":0.1, "study_noise":1}' \
