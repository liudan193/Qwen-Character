set -x

export PYTHONUNBUFFERED=1
mkdir ./logs
mkdir -p logs

PROJECT_NAME="RM-NLHF"
EXP_NAME="exp-ds_r1_distill_qwen_7b-rm_nlhf-no-metarm"
FULL_NAME=${PROJECT_NAME}_${EXP_NAME}
INIT_CKPT_POLICY_MODEL="/path/to/DeepSeek-R1-Distill-Qwen-7B"
INIT_CKPT_REWARD_MODEL="/path/to/Cold-Start-MetaRM-FSDP-Qwen-7B"
train_files="['/path/to/rm_nlhf_train_rl_helpsteer3.parquet']"  # you should filter the data to keep only the HelpSteer3 portion
TEST="/path/to/helpsteer3_validation_rl_modified_human_critique_rl_split_fix3.parquet"

nnodes=4
nproc_per_node=8

echo "RANK: $RANK"
echo "$RANK == 0"
echo "PET_MASTER_PORT: $PET_MASTER_PORT"

if [[ "$RANK" == 0 ]]; then
    ray start --head --port=$PET_MASTER_PORT
    echo "Ray head started on port: $PET_MASTER_PORT"
    sleep 120s

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${TRAIN} \
  data.val_files=${TEST} \
  data.train_batch_size=256 \
  data.max_prompt_length=6144 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=${INIT_CKPT_POLICY_MODEL} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=43008 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy="fsdp2" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_num_batched_tokens=43008 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.use_kl_in_reward=False \
  reward_model.reward_manager=batch \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="/path/to/Qwen-Character/Character-GenRM-NLHF/verl/recipe/rm_nlhf/reward_function.py" \
  custom_reward_function.name=reward_function_rm_nlhf_batch \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXP_NAME} \
  trainer.default_local_dir="/path/to/checkpoints/${FULL_NAME}" \
  trainer.rollout_data_dir="/path/to/rollouts/${FULL_NAME}/train" \
  trainer.validation_data_dir="/path/to/rollouts/${FULL_NAME}/validation" \
  trainer.resume_mode="auto" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=${nproc_per_node} \
  trainer.nnodes=${nnodes} \
  trainer.save_freq=40 \
  trainer.test_freq=20 \
  trainer.total_epochs=20 2>&1 | tee ./logs/${FULL_NAME}.rank_${RANK}.log

else
    echo "Worker node, rank: $RANK"
    sleep 90s
    ray start --address="$MASTER_ADDR:$PET_MASTER_PORT"
    echo "Ray worker joined head at $MASTER_ADDR:$PET_MASTER_PORT"
    sleep infinity
fi