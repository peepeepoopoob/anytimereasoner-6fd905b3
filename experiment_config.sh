#!/bin/bash
# Method configuration — ARM: AnytimeReasoner-uniform.
#
# Sampled budgets with verifiable DENSE rewards (4 budget supports), a UNIFORM
# prior over budgets for the thinking policy, BRPO variance reduction, and a
# decoupled summary policy trained over a uniform budget distribution
# (n_summary=4). Target: beat GRPO on both final and anytime accuracy at 600 steps.

# --- shared (identical across arms for a fair comparison) ---
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export MAX_GEN_LEN=4000
export MAX_RESPONSE_LENGTH=4224
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=32
export ROLLOUT_N=8
export TOTAL_STEPS=600
export TEST_FREQ=40

# --- method-specific: AnytimeReasoner-uniform ---
export EXP_NAME="AR-uniform"
export N_SUMMARY=4
export SUMMARY_METHOD="brpo"
export N_BUDGET_SUPPORT=4
export BUDGET_PROBS="uniform"
export VARIANCE_REDUCTION="brpo"
