python -m domainbed.scripts.sweep launch\
    --data_dir=./domainbed/data/\
    --output_dir=./results/cmnist_noise_levels/\
    --command_launcher multi_gpu\
    --algorithms IRM ERM GroupDRO VREx Mixup    \
    --datasets CMNISTMod  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2          \
    --hparams '{"flip_prob":0.25, "study_noise":1}' \
    --skip_confirmation


python -m domainbed.scripts.sweep launch\
    --data_dir=./domainbed/data/\
    --output_dir=./results/cmnist_noise_levels/\
    --command_launcher multi_gpu\
    --algorithms IRM ERM GroupDRO VREx Mixup   \
    --datasets CMNISTMod  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2          \
    --hparams '{"flip_prob":0.2, "study_noise":1}' \
    --skip_confirmation


python -m domainbed.scripts.sweep launch\
    --data_dir=./data/\
    --output_dir=./results/cmnist_noise_levels/\
    --command_launcher multi_gpu\
    --algorithms IRM ERM GroupDRO VREx Mixup   \
    --datasets CMNISTMod  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2          \
    --hparams '{"flip_prob":0.15, "study_noise":1}' \
    --skip_confirmation

python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/cmnist_noise_levels/\
    --command_launcher multi_gpu\
    --algorithms IRM ERM GroupDRO VREx Mixup    \
    --datasets CMNISTMod  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2          \
    --hparams '{"flip_prob":0.1, "study_noise":1}' \
    --skip_confirmation


python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/cmnist_noise_levels/\
    --command_launcher multi_gpu\
    --algorithms IRM ERM GroupDRO VREx Mixup    \
    --datasets CMNISTMod  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2          \
    --hparams '{"flip_prob":0.05, "study_noise":1}' \
    --skip_confirmation


python -m domainbed.scripts.sweep launch\
    --data_dir=~/data/\
    --output_dir=./results/cmnist_noise_levels/\
    --command_launcher multi_gpu\
    --algorithms IRM ERM GroupDRO VREx Mixup   \
    --datasets CMNISTMod  \
    --n_hparams 20       \
    --n_trials 3       \
    --test_envs 2          \
    --skip_confirmation