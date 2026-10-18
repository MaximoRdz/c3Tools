# Deploy Local LLM in C3 Cluster

## Quickstart
```bash
export LUSTRE=/lustre/uc3m_a0/dynamic/maxrodri/language-models
mkdir -p "$LUSTRE"/{apptainer-cache,apptainer-tmp,images,models,cache,webui-data,logs}

export APPTAINER_CACHEDIR="$LUSTRE/apptainer-cache"
export APPTAINER_TMPDIR="$LUSTRE/apptainer-tmp"
export HF_HOME="$LUSTRE/cache/huggingface"

apptainer pull images/vllm.sif \
  docker://vllm/vllm-openai:v0.10.2

apptainer pull images/open-webui.sif \
  docker://ghcr.io/open-webui/open-webui:v0.6.26

apptainer exec \
  --bind "$LUSTRE/models:/models" \
  --bind "$LUSTRE/cache:/cache" \
  --env HF_HOME="$LUSTRE/cache/huggingface" \
  --env HF_TOKEN="hf_your-token" \
  "$LUSTRE/images/vllm.sif" \
  hf download Qwen/Qwen3-14B \
    --local-dir "$LUSTRE/models/Qwen3-14B"
```


it might happen that we need a secret key env for the webui:

```bash
umask 077
[ -s webui-secret-key ] || openssl rand -hex 32 > webui-secret-key
cat webui-secret-key 
> 821543501dfc5....f8cd9d45bacd8d50fc
```

not only this but working via uc3m proxy...
in webui apptainer:
```bash
--env NO_PROXY=127.0.0.1,localhost \
--env no_proxy=127.0.0.1,localhost \
```

### Llama 3.1
awaiting for permission
```bash
apptainer exec \
    --bind "$LUSTRE/models:/models" \
    --bind "$LUSTRE/cache:/cache" \
    --env HF_HOME="$LUSTRE/cache/huggingface" \
    --env HF_TOKEN="hf_your-token" \
    "$LUSTRE/images/vllm.sif" \
    hf download meta-llama/Llama-3.1-8B \
    --local-dir "$LUSTRE/models/Llama3.1-8B"
```
