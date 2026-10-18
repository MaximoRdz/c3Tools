#!/usr/bin/env bash
#SBATCH --job-name=local-llm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-large
#SBATCH --output=logs/llm-%j.log

set -Eeuo pipefail

ROOT="/lustre/uc3m_a0/dynamic/maxrodri/language-models"

export APPTAINER_CACHEDIR="$ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$ROOT/apptainer-tmp"
export TMPDIR="$ROOT/apptainer-tmp"
export HF_HOME="$ROOT/cache/huggingface"

VLLM_IMAGE="$ROOT/images/vllm.sif"
WEBUI_IMAGE="$ROOT/images/open-webui.sif"
ROUTER_SCRIPT="$ROOT/scripts/model-router.py"

API_PORT=8000
UI_PORT=8080

ROUTER_PID=""
WEBUI_PID=""

mkdir -p \
    "$APPTAINER_CACHEDIR" \
    "$APPTAINER_TMPDIR" \
    "$HF_HOME" \
    "$ROOT/logs" \
    "$ROOT/webui-data" \
    "$ROOT/run" \
    "$ROOT/scripts"

cd "$ROOT"

# The default model must exist before the service starts.
for path in \
    "$ROOT/models/Qwen3-14B" \
    "$VLLM_IMAGE" \
    "$WEBUI_IMAGE" \
    "$ROUTER_SCRIPT" \
    "$ROOT/webui-secret-key"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Missing required path: $path"
        exit 1
    fi
done

if [[ ! -s "$ROOT/webui-secret-key" ]]; then
    echo "ERROR: $ROOT/webui-secret-key is empty."
    echo "Create a persistent secret key before starting Open WebUI."
    exit 1
fi

cleanup() {
    echo "Stopping services..."

    # Stop the UI first, then the router. The router is responsible
    # for stopping its vLLM child process during graceful shutdown.
    if [[ -n "${WEBUI_PID:-}" ]]; then
        kill "$WEBUI_PID" 2>/dev/null || true
        wait "$WEBUI_PID" 2>/dev/null || true
        WEBUI_PID=""
    fi

    if [[ -n "${ROUTER_PID:-}" ]]; then
        kill -TERM "$ROUTER_PID" 2>/dev/null || true

        # Give the router/container time to stop vLLM cleanly.
        for _ in $(seq 1 15); do
            if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$ROUTER_PID" 2>/dev/null; then
            echo "Router did not stop in time; sending SIGKILL."
            kill -KILL "$ROUTER_PID" 2>/dev/null || true
        fi

        wait "$ROUTER_PID" 2>/dev/null || true
        ROUTER_PID=""
    fi
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Host: $(hostname -f)"
echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Root: $ROOT"
echo "Starting model router (which starts the default Qwen model)..."

# The router listens on port 8000 and starts the active vLLM process
# internally on port 8001. Model paths are mounted at /models.
apptainer exec --nv \
    --bind "$ROOT/models:/models" \
    --bind "$ROOT/cache:/cache" \
    --bind "$ROOT/logs:/logs" \
    --bind "$ROOT/scripts:/scripts" \
    --env HF_HOME=/cache/huggingface \
    --env NO_PROXY=127.0.0.1,localhost \
    --env no_proxy=127.0.0.1,localhost \
    "$VLLM_IMAGE" \
    python3 /scripts/model-router.py \
    > "$ROOT/logs/router-$SLURM_JOB_ID.log" 2>&1 &

ROUTER_PID=$!

echo "Waiting for the router and default model..."

ROUTER_READY=0
for i in $(seq 1 180); do
    if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
        echo "ERROR: Model router exited."
        tail -n 100 "$ROOT/logs/router-$SLURM_JOB_ID.log" || true
        exit 1
    fi

    if curl --noproxy '*' -fsS \
        "http://127.0.0.1:$API_PORT/health" \
        -o "$ROOT/run/router-health.json"; then
        if grep -q '"status"[[:space:]]*:[[:space:]]*"ok"' \
            "$ROOT/run/router-health.json"; then
            ROUTER_READY=1
            break
        fi
    fi

    sleep 10
done

if [[ "$ROUTER_READY" -ne 1 ]]; then
    echo "ERROR: Router/default model did not become ready in time."
    echo "Router log:"
    tail -n 100 "$ROOT/logs/router-$SLURM_JOB_ID.log" || true
    echo "Qwen vLLM log:"
    tail -n 100 "$ROOT/logs/vllm-Qwen3-14B.log" 2>/dev/null || true
    exit 1
fi

echo "Model router and default model are ready."

WEBUI_DATA="$ROOT/webui-data"
mkdir -p "$WEBUI_DATA"

echo "Starting Open WebUI..."

apptainer run \
    --pwd /app/backend \
    --bind "$WEBUI_DATA:/app/backend/data" \
    --env HOST=127.0.0.1 \
    --env PORT="$UI_PORT" \
    --env OPENAI_API_BASE_URL="http://127.0.0.1:$API_PORT/v1" \
    --env OPENAI_API_KEY=local-only \
    --env WEBUI_AUTH=true \
    --env "WEBUI_SECRET_KEY=$(cat "$ROOT/webui-secret-key")" \
    --env NO_PROXY=127.0.0.1,localhost \
    --env no_proxy=127.0.0.1,localhost \
    "$WEBUI_IMAGE" \
    > "$ROOT/logs/webui-$SLURM_JOB_ID.log" 2>&1 &

WEBUI_PID=$!

echo "Waiting for Open WebUI..."

UI_READY=0
for i in $(seq 1 30); do
    if ! kill -0 "$WEBUI_PID" 2>/dev/null; then
        echo "ERROR: Open WebUI exited."
        tail -n 100 "$ROOT/logs/webui-$SLURM_JOB_ID.log" || true
        exit 1
    fi

    if curl --noproxy '*' -fsS \
        "http://127.0.0.1:$UI_PORT/" \
        >/dev/null 2>&1; then
        UI_READY=1
        break
    fi

    sleep 2
done

if [[ "$UI_READY" -ne 1 ]]; then
    echo "ERROR: Open WebUI did not become ready."
    tail -n 100 "$ROOT/logs/webui-$SLURM_JOB_ID.log" || true
    exit 1
fi

echo
echo "========================================"
echo "LLM SERVICE READY"
echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname -f)"
echo "Open WebUI port: $UI_PORT"
echo "Model router port: $API_PORT"
echo "Internal vLLM port: 8001"
echo "========================================"
echo
echo "Keep this Slurm allocation running."
echo "Router log: $ROOT/logs/router-$SLURM_JOB_ID.log"
echo "WebUI log:  $ROOT/logs/webui-$SLURM_JOB_ID.log"
echo "Model list: curl --noproxy '*' http://127.0.0.1:$API_PORT/v1/models"

# Keep the Slurm job alive while both long-running services are alive.
while kill -0 "$ROUTER_PID" 2>/dev/null &&
      kill -0 "$WEBUI_PID" 2>/dev/null; do
    sleep 10
done

echo "ERROR: Router or Open WebUI stopped."
exit 1
