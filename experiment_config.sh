#!/bin/bash
# Method configuration for the AnytimeReasoner minimal reproduction.
#
# Sibling experiment nodes override THIS FILE only (run.sh is shared infra).
# Default here is the GRPO baseline; the AnytimeReasoner-uniform sibling sets
# the BRPO / multi-budget / uniform-prior knobs.

# --- shared (same for every arm, for a fair comparison) ---
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export MAX_GEN_LEN=4000
export MAX_RESPONSE_LENGTH=4224
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=32
export ROLLOUT_N=8
export TOTAL_STEPS=600
export TEST_FREQ=40

# --- method-specific (GRPO baseline) ---
export EXP_NAME="GRPO"
export N_SUMMARY=1
export SUMMARY_METHOD="grpo"
export N_BUDGET_SUPPORT=1
export BUDGET_PROBS="base"
export VARIANCE_REDUCTION="v2only"
