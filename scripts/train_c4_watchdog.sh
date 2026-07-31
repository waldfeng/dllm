#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"
CHECKPOINT="$ROOT_DIR/datasets/checkpoints/lastest_weight.pt"
LOG_FILE="$ROOT_DIR/datasets/logs/lastest_weight-training.log"
NAN_LOG_FILE="$ROOT_DIR/datasets/logs/lastest_weight-training-nan.log"
READY_FILE="$ROOT_DIR/datasets/logs/lastest_weight-training.ready"
BATCH_SIZE="${BATCH_SIZE:-1136}"
LENGTH_WEIGHT="${LENGTH_WEIGHT:-1e-3}"
MEMORY_LIMIT_PERCENT="${MEMORY_LIMIT_PERCENT:-90}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
RESTART_DELAY="${RESTART_DELAY:-5}"

mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$LOG_FILE")"
if [[ ! -f "$CHECKPOINT" ]]; then
    printf 'checkpoint does not exist; starting a fresh run: %s\n' "$CHECKPOINT" >&2
fi

TRAIN_PID=""
STOP_REQUESTED=0

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        request_graceful_stop
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup EXIT INT TERM

request_graceful_stop() {
    local rank_pid
    if [[ -n "$TRAIN_PID" ]]; then
        kill -USR1 -- "-$TRAIN_PID" 2>/dev/null || true
    fi
    for rank_pid in $(pgrep -P "$TRAIN_PID" 2>/dev/null || true); do
        kill -USR1 "$rank_pid" 2>/dev/null || true
    done
}

gpu_over_limit() {
    local used total percent
    while IFS=',' read -r used total; do
        used="${used//[[:space:]]/}"
        total="${total//[[:space:]]/}"
        [[ -z "$used" || -z "$total" || "$total" -eq 0 ]] && continue
        percent=$((used * 100 / total))
        if (( percent >= MEMORY_LIMIT_PERCENT )); then
            printf 'GPU memory limit reached: %s%% (%s/%s MiB)\n' "$percent" "$used" "$total" >&2
            return 0
        fi
    done < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    return 1
}

wait_for_gpu_release() {
    local used
    while true; do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | awk 'BEGIN { busy=0 } { if ($1 > 0) busy=1 } END { print busy }')"
        [[ "$used" == "0" ]] && return 0
        sleep 2
    done
}

while true; do
    STOP_REQUESTED=0
    rm -f "$READY_FILE"
    RESUME_ARGS=()
    if [[ -f "$CHECKPOINT" ]]; then
        RESUME_ARGS=(--resume-checkpoint "$CHECKPOINT")
    fi
    setsid "$TORCHRUN" --standalone --nproc_per_node=2 \
        scripts/train_c4.py \
        --distributed --device cuda \
        --dataset-dir datasets/c4_qwen2.5 \
        --epochs 1 --batch-size "$BATCH_SIZE" \
        --length-weight "$LENGTH_WEIGHT" \
        --lr 7.5e-5 --warmup-steps 128 \
        --checkpoint "$CHECKPOINT" \
        "${RESUME_ARGS[@]}" \
        --log-file "$LOG_FILE" \
        --nan-log-file "$NAN_LOG_FILE" \
        --ready-file "$READY_FILE" \
        --checkpoint-hours 1 &
    TRAIN_PID=$!

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"
        if [[ -f "$READY_FILE" ]] && gpu_over_limit; then
            STOP_REQUESTED=1
            request_graceful_stop
            break
        fi
    done

    train_status=0
    wait "$TRAIN_PID" 2>/dev/null || train_status=$?
    TRAIN_PID=""

    if (( STOP_REQUESTED == 0 && train_status == 0 )); then
        exit 0
    fi

    if (( STOP_REQUESTED != 0 )); then
        wait_for_gpu_release
        if (( BATCH_SIZE <= MIN_BATCH_SIZE )); then
            printf 'cannot reduce batch size below %s\n' "$MIN_BATCH_SIZE" >&2
            exit 1
        fi
        BATCH_SIZE=$((BATCH_SIZE * 2 / 3))
        (( BATCH_SIZE < MIN_BATCH_SIZE )) && BATCH_SIZE="$MIN_BATCH_SIZE"
        printf 'restart reason=memory_limit status=%s checkpoint=%s per-GPU batch_size=%s\n' \
            "$train_status" "$CHECKPOINT" "$BATCH_SIZE" | tee -a "$LOG_FILE"
    else
        wait_for_gpu_release
        printf 'restart reason=training_error status=%s checkpoint=%s per-GPU batch_size=%s\n' \
            "$train_status" "$CHECKPOINT" "$BATCH_SIZE" | tee -a "$LOG_FILE" >&2
    fi
    sleep "$RESTART_DELAY"
done
