import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

MODEL_PATHS = {
    "Qwen3-14B": "/models/Qwen3-14B",
    "Llama3.1-8B": "/models/Llama3.1-8B",
    "DeepSeek-R1-Distill-Qwen-14B":
        "/models/DeepSeek-R1-Distill-Qwen-14B",
}

DEFAULT_MODEL = "Qwen3-14B"
VLLM_PORT = 8001
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"
ROUTER_PORT = 8000
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    name: path for name, path in MODEL_PATHS.items()
    if os.path.isdir(path)
}

SYSTEM_CONTEXT_PATH = Path("/scripts/system-prompt.txt")
SYSTEM_CONTEXT = SYSTEM_CONTEXT_PATH.read_text().strip()

app = FastAPI()
process = None
process_log = None
active_model = None
switch_lock = asyncio.Lock()
http = httpx.AsyncClient(
    trust_env=False,
    timeout=None,
)


async def stop_model():
    global process, process_log, active_model

    p = process
    process = None
    active_model = None

    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            await asyncio.to_thread(p.wait, timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.to_thread(p.wait)

    if process_log is not None:
        process_log.close()
        process_log = None


async def start_model(name):
    global process, process_log, active_model

    if name not in MODELS:
        raise ValueError(f"Model is not installed: {name}")

    model_path = MODELS[name]

    print(f"Starting model {name}: {model_path}", flush=True)

    process_log = open(
        LOG_DIR / f"vllm-{name}.log", "a", buffering=1
    )

    process = subprocess.Popen(
        [
            "vllm", "serve", model_path,
            "--served-model-name", name,
            "--host", "127.0.0.1",
            "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
        ],
        stdout=process_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    for _ in range(600):
        if process.poll() is not None:
            await stop_model()
            raise RuntimeError(
                f"vLLM exited during startup; inspect "
                f"/logs/vllm-{name}.log"
            )

        try:
            response = await http.get(
                f"{VLLM_URL}/models", timeout=3
            )
            if response.status_code == 200:
                data = response.json().get("data", [])
                if any(item.get("id") == name for item in data):
                    active_model = name
                    print(f"Model ready: {name}", flush=True)
                    return
        except (httpx.HTTPError, ValueError):
            pass

        await asyncio.sleep(2)

    await stop_model()
    raise RuntimeError(
        f"Timed out loading {name}; inspect /logs/vllm-{name}.log"
    )


async def ensure_model(name):
    if name not in MODELS:
        raise ValueError(
            f"Unknown or unavailable model: {name}. "
            f"Available: {', '.join(MODELS)}"
        )

    if (
        active_model == name
        and process is not None
        and process.poll() is None
    ):
        return

    previous = active_model
    await stop_model()

    try:
        await start_model(name)
    except Exception:
        if previous and previous in MODELS and previous != name:
            try:
                await start_model(previous)
            except Exception as exc:
                print(f"Could not restore {previous}: {exc}", flush=True)
        raise


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_model": active_model,
        "available_models": list(MODELS),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
            for name in MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON request"}},
            status_code=400,
        )

    name = payload.get("model", DEFAULT_MODEL)
    messages = payload.get("messages", [])

    if isinstance(messages, list):
        messages.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_CONTEXT,
            },
        )

    await switch_lock.acquire()

    try:
        await ensure_model(name)

        upstream_request = http.build_request(
            "POST",
            f"{VLLM_URL}/chat/completions",
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
        )
        upstream = await http.send(upstream_request, stream=True)

        headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in {
                "content-length", "content-encoding",
                "transfer-encoding", "connection",
                "keep-alive",
            }
        }

        if payload.get("stream", False):
            async def stream_body():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()
                    switch_lock.release()

            return StreamingResponse(
                stream_body(),
                status_code=upstream.status_code,
                headers=headers,
            )

        content = await upstream.aread()
        status = upstream.status_code
        await upstream.aclose()
        switch_lock.release()

        return Response(
            content=content,
            status_code=status,
            headers=headers,
        )

    except ValueError as exc:
        switch_lock.release()
        return JSONResponse(
            {"error": {"message": str(exc)}},
            status_code=400,
        )
    except Exception as exc:
        switch_lock.release()
        print(f"Request failed: {exc}", flush=True)
        return JSONResponse(
            {"error": {"message": str(exc)}},
            status_code=502,
        )


@app.on_event("startup")
async def startup():
    await ensure_model(DEFAULT_MODEL)


@app.on_event("shutdown")
async def shutdown():
    await stop_model()
    await http.aclose()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=ROUTER_PORT)

