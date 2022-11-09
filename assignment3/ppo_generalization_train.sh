# Example script to run a batch of PPO generalization experiments

# 1M steps is enough.
# I set "--restore" here because I already run a batch of 1M steps experiments.
# And I want to finetune the trained agents so each of them is trained with 2M steps.

for num in 1 5 10 20 50 100; do
  python train_ppo.py \
  --env-id MetaDrive-Tut-${num}Env-v0 \
  --log-dir MetaDrive-Tut-${num}Env-v0 \
  --num-envs 10 \
  --max-steps 1000000 \
  --restore \
  > ppo_metadrive_${num}env_train.log 2>&1 &
done


# Run this command to overwatch the training progress:
# watch tail -n 5 ppo*train.log
