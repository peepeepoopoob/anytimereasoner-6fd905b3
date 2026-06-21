#!/bin/bash
# Minimal end-to-end reproduction of AnytimeReasoner (arXiv 2505.13438).
#
# This single script: installs deps, prepares the DeepScaler data, and runs
# verl PPO training, emitting anytime + final accuracy to
# .openresearch/artifacts/EVAL.md.
#
# Method flags (GRPO vs AnytimeReasoner-uniform) live in experiment_config.sh,
# which sibling experiment nodes override. This script itself is shared infra.
set -euo pipefail
set -x

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ART_DIR="$REPO_ROOT/.openresearch/artifacts"
mkdir -p "$ART_DIR"

############################################
# 1. Dependencies
############################################
# nvidia-smi sanity
nvidia-smi || true
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

pip install -q --upgrade pip
# Pin the stack the repo was built against.
pip install -q torch==2.6.0
pip install -q -e ./verl
pip install -q vllm==0.8.1
pip install -q flash-attn==2.7.4.post1 --no-build-isolation || \
  pip install -q flash-attn --no-build-isolation
pip install -q -e .
pip install -q antlr4-python3-runtime==4.9.3 tensordict==0.6.2 transformers==4.50.1 \
  hf_transfer latex2sympy2 pylatexenc tabulate wandb

############################################
# 2. Data preparation (from bundled JSON, no download)
############################################
if [ ! -f "$HOME/deepscaler/data/train.parquet" ]; then
  python3 anytime_reasoner/scripts/data/deepscaler_dataset.py
fi
ls -la "$HOME/deepscaler/data/"

############################################
# 3. Method configuration (overridden by siblings)
############################################
source "$REPO_ROOT/experiment_config.sh"

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

# Minimal-repro shared knobs.
MAX_GEN_LEN="${MAX_GEN_LEN:-4000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4224}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-600}"
TEST_FREQ="${TEST_FREQ:-40}"
N_GPUS="${N_GPUS:-$NUM_GPUS}"

echo "=== Run config ==="
echo "EXP_NAME=$EXP_NAME"
echo "METHOD: n_summary=$N_SUMMARY summary_method=$SUMMARY_METHOD n_budget_support=$N_BUDGET_SUPPORT budget_probs=$BUDGET_PROBS variance_reduction=$VARIANCE_REDUCTION"
echo "max_gen_len=$MAX_GEN_LEN total_steps=$TOTAL_STEPS n_gpus=$N_GPUS rollout_n=$ROLLOUT_N"

############################################
# 4. Train
############################################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/train.parquet \
    data.val_files=[$HOME/deepscaler/data/aime.parquet,$HOME/deepscaler/data/amc.parquet] \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.attn_implementation=flex_attention \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.n_summary=$N_SUMMARY \
    actor_rollout_ref.rollout.max_gen_len=$MAX_GEN_LEN \
    actor_rollout_ref.rollout.n_budget_support=$N_BUDGET_SUPPORT \
    actor_rollout_ref.rollout.budget_probs="$BUDGET_PROBS" \
    actor_rollout_ref.rollout.summary_method="$SUMMARY_METHOD" \
    actor_rollout_ref.rollout.variance_reduction="$VARIANCE_REDUCTION" \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='anytime-reasoning-minimal' \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=True \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.val_generations_to_log_to_wandb=5 \
    trainer.total_epochs=1

echo "=== Training finished. EVAL.md: ==="
cat "$ART_DIR/EVAL.md" || true
