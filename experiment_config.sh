#!/bin/bash
# Method configuration — ARM: GRPO baseline (control).
#
# Single full-budget thinking policy, sparse end-of-trace reward, no dense
# rewards, no BRPO variance reduction. This is the baseline we aim to beat.

# --- shared (identical across arms for a fair comparison) ---
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export MAX_GEN_LEN=4000
export MAX_RESPONSE_LENGTH=4224
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=32
export ROLLOUT_N=8
export TOTAL_STEPS=600
export TEST_FREQ=40

# --- method-specific: GRPO ---
export EXP_NAME="GRPO"
export N_SUMMARY=1
export SUMMARY_METHOD="grpo"
export N_BUDGET_SUPPORT=1
export BUDGET_PROBS="base"
export VARIANCE_REDUCTION="v2only"
