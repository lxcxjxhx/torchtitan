# Distributed CPU KV cache

Titan RL uses vLLM's
[`MooncakeStoreConnector`](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
to share prompt and optional decode KV across generators through a distributed
CPU DRAM pool. Generators keep their local GPU cache; misses recompute from the
request's full token history.

Store manages CPU capacity and eviction; vLLM manages GPU blocks. The shared pool
works without prefill/decode disaggregation. The separate
[`MooncakeConnector`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)
handles point-to-point prefill/decode transfers.

## Setup

Install a compatible Mooncake SDK on each generator host (tested:
`mooncake-transfer-engine==0.3.13.post1`). RDMA requires RNICs, GPU memory
registration, and the SDK's native libraries. Start a reachable Master for metadata:

```bash
mooncake_master --rpc_port=50051
```

Set `MOONCAKE_CONFIG_PATH` in every generator process to a local JSON file:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "MASTER_HOST:50051",
  "global_segment_size": "8GB",
  "local_buffer_size": "1GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `global_segment_size` is DRAM **per worker**: four workers at 8 GB use 32 GB.
  Leave room for weight-prefetch buffers and other host processes.
- `enable_offload` controls the **disk tier**; CPU offload works with `false`.

Host-specific settings used in the RDMA test:

- Set `expandable_segments:False` in both `PYTORCH_ALLOC_CONF` and
  `PYTORCH_CUDA_ALLOC_CONF` before importing PyTorch. Expandable allocations
  have not been validated.
- Set `WITH_NVIDIA_PEERMEM=0` for the tested SDK's DMA-BUF path.
- Set `VLLM_HOST_IP` only if auto-detection fails. Use a routable address for
  multiple hosts; `127.0.0.1` was used for the single-host test.

## Generator configuration

```python
from vllm.config import KVTransferConfig

from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter

engine = VLLMEngine.Config(
    kv_transfer_config=KVTransferConfig(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cache_prefix": "unique-run-and-model-id",
            "save_decode_cache": True,
        },
    ),
)
generator_router = InterGeneratorRouter.Config(hot_swap=False)
```

- Use a shared, non-empty `cache_prefix` that is **new for each run or restart**,
  including checkpoint resume. Generators sharing KV must match model, dtype,
  KV layout, runtime versions, and hash settings, including any `PYTHONHASHSEED`
  override. Prefix caching and hashing otherwise use vLLM defaults.
- `kv_both` enables reads and writes; `save_decode_cache=True` also saves generated
  KV. Other native roles and save options pass through.

## Weight and cache lifecycle

- Load trainer weights before the first request. Titan includes the **loaded
  policy version** in cache keys so weight updates cannot reuse stale KV.
- Keep `generator_router.hot_swap=False` to drain model calls before weight
  updates, and `reset_kv_cache_on_model_state_dict_load=False`: asynchronous saves
  can still hold GPU blocks after a response. The engine drains pending transfers.
- Old policy entries remain available until Store evicts them. Do not call
  `reset_prefix_cache(reset_connector=True)` on a replica; it can clear the
  shared Store globally.

## Supported configurations

- **Multi-GPU:** Titan's TP and DP/EP layouts are supported. TP ranks coordinate
  KV lookup and completion using shared logical request IDs while retaining
  rank-local vLLM IDs; DP groups coordinate progress. Use `num_generators` for
  independent replicas.
- **Connector:** direct `MooncakeStoreConnector` only; multi-GPU mode requires
  vLLM's built-in implementation. `kv_transfer_config=None` disables it.
- **Sessions:** `session_kv_holder` retains local GPU KV; releasing it does not
  remove shared Store entries.
- **Unsupported:** `MultiConnector`, other connectors, batch-invariant mode,
  standalone `generate.py`, and partial-block prefix hits for hybrid producers.
  Use full-block prefix matching for hybrid producers.

## Validation

Select idle RDMA-capable GPUs with `CUDA_VISIBLE_DEVICES` and run from the repo
root. The full matrix needs eight GPUs; use `-k 'not tp2 and not dp2'` for two.

```bash
RUN_MOONCAKE_TEST=1 pytest -s tests/rl/unit_tests/gpu/test_mooncake_store.py
```

Tests cover shared KV reuse, policy-version isolation, and TP/DP/EP coordination.
Multi-node operation, CUDA graphs, and performance remain unvalidated.

See the [architecture overview](https://vllm.ai/blog/2026-05-06-mooncake-store).
