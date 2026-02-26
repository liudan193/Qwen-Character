set -x

export PYTHONUNBUFFERED=1
mkdir -p logs

GRM_HOST="http://todo:8011/v1"
GRM_NAME="grm_rmnlhf"

PROJECT_NAME="RM-NLHF"
EXP_NAME="exp-ds_r1_distill_qwen_7b-rm_nlhf_as_grm"
FULL_NAME=${PROJECT_NAME}_${EXP_NAME}
INIT_CKPT_POLICY_MODEL="/path/to/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TRAIN="/path/to/train.parquet"
TEST="/path/to/test.parquet"

nnodes=1
nproc_per_node=8

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${TRAIN} \
  data.val_files=${TEST} \
  data.train_batch_size=128 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=${INIT_CKPT_POLICY_MODEL} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy="fsdp2" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=36864 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="/path/to/RM-NLHF/verl/recipe/rm_nlhf/reward_function.py" \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=pairwise_grm \
  +reward_model.reward_kwargs.grm_host=${GRM_HOST} \
  +reward_model.reward_kwargs.grm_name=${GRM_NAME} \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXP_NAME} \
  trainer.default_local_dir="/path/to/checkpoints/${FULL_NAME}" \
  trainer.rollout_data_dir="/path/to/rollouts/${FULL_NAME}/train" \
  trainer.validation_data_dir="/path/to/rollouts/${FULL_NAME}/validation" \
  trainer.resume_mode="auto" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=${nproc_per_node} \
  trainer.nnodes=${nnodes} \
  trainer.save_freq=20 \
  trainer.test_freq=999999 \
  trainer.total_epochs=1 2>&1 | tee ./logs/${FULL_NAME}.rank_${RANK}.log