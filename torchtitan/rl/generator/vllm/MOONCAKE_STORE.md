# Distributed CPU KV cache

Titan RL uses vLLM's
[`MooncakeStoreConnector`](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
to share prompt and optional decode KV through a CPU DRAM pool. Each generator keeps
its local GPU cache and can load shared KV after a router repin. Cache misses
recompute from the request's full token history.

Mooncake Master tracks metadata; Store manages CPU capacity and eviction;
Transfer Engine moves KV; vLLM manages GPU blocks. This integration requires no
prefill/decode disaggregation. The separate
[`MooncakeConnector`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)
provides point-to-point prefill/decode transfers, not a shared cache pool.

## Setup

Install a Mooncake SDK compatible with your vLLM build on each generator host.
The smoke test used `mooncake-transfer-engine==0.3.13.post1`. RDMA requires working
RNICs, GPU memory registration, and the SDK's native libraries.

Start a reachable Master outside the RL job:

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

`global_segment_size` is DRAM contributed **per worker**: four workers at 8 GB
use 32 GB of their host's DRAM. Reserve memory for weight-prefetch buffers and other
processes. `enable_offload` controls the disk tier; leave it `false` for CPU-only
KV offload. vLLM loads and validates the Store configuration.

The devgpu test used these environment overrides; adapt them to your host:

- `expandable_segments:False` in both `PYTORCH_ALLOC_CONF` and
  `PYTORCH_CUDA_ALLOC_CONF`, set before importing PyTorch to override Titan's
  `True` defaults. Expandable allocations have not been validated.
- `WITH_NVIDIA_PEERMEM=0` for the tested SDK's NVIDIA DMA-BUF registration
  path. Other SDKs or `nvidia-peermem` deployments may differ.
- `VLLM_HOST_IP=127.0.0.1` for the single-host test. Override auto-detection
  only when needed; multi-host deployments require a routable address per host.

## Generator configuration

Use native vLLM configuration in the RL recipe:

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

- Share one `cache_prefix` across compatible generators, and choose a **new prefix
  for every run or restart**, including checkpoint resume. Titan checks only
  that it is non-empty; the caller ensures uniqueness and matching model, dtype,
  and KV layout.
- `kv_both` enables reads and writes; `save_decode_cache=True` also saves generated
  KV blocks. These are recommended for multi-turn reuse, not enforced. Other
  native roles and save options pass through; decode saving defaults to `False`
  in the tested build.
- Prefix caching and hashing use vLLM defaults (enabled for supported models,
  `sha256` in the tested build). Keep runtime versions, hash settings, and any
  `PYTHONHASHSEED` override consistent across generators sharing KV.

## Weight and cache lifecycle

Trainer weights must load before the first request. Titan salts cache hashes with
each generator's **loaded policy version**, even if the rollout began at an older
version. The controller waits for all generators to load a snapshot before the
trainer publishes the next, so a version identifies the same weights everywhere.

Set `generator_router.hot_swap=False`: updates drain in-flight model calls,
not entire multi-turn rollouts. Loading weights with unfinished requests raises.
Keep `reset_kv_cache_on_model_state_dict_load=False`; asynchronous saves may still
hold GPU blocks after a completion is returned. The engine continues stepping
until those transfers finish and release their references.

Versioned keys isolate old KV while leaving it available to replicas still using
those weights; Store can evict it as needed. Do not call
`reset_prefix_cache(reset_connector=True)` on one generator: it can clear the
shared Store globally.

## Supported configurations

- **One GPU per generator (`TP=DP=EP=1`).** Scale with `num_generators`.
  Multi-GPU support needs coordinated lookup results and transfer completion
  across replicated schedulers.
- **Direct `MooncakeStoreConnector` only.** `MultiConnector` and other connectors
  are rejected; `kv_transfer_config=None` disables the connector.
- With prefix caching enabled, Store producers reject **hybrid partial-block
  hits**: vLLM can skip finish-time tail saves and leave GPU blocks pinned. Use
  full-block matching for hybrid models; a `prefix_match_unit` finer than the
  Mamba block size can trigger this path. Consumers and full-attention models
  are unaffected.
- `session_kv_holder` can retain local GPU KV. Releasing a session drops its held
  GPU references, not shared Store entries; the connector does not manage sessions.
- Batch-invariant mode is unsupported because it requires cache resets on weight
  load. Standalone `generate.py` still needs checkpoint namespacing and pending-save
  draining before it can use Store.

## Validation

Select two idle RDMA-capable GPUs with `CUDA_VISIBLE_DEVICES` (numeric IDs or UUIDs)
and run from the repository root:

```bash
RUN_MOONCAKE_TEST=1 pytest -s tests/rl/unit_tests/gpu/test_mooncake_store.py
```

The test starts a temporary Master and two generators. It checks cross-generator
and next-turn reuse, policy-version isolation, prompt/decode saving, connector
roles, and session retention. It verifies GPU selection and covers 256-token
blocks. This is functional validation; multi-node performance and session-driven
Store eviction remain follow-ups.

See the [Store setup guide](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
and [architecture overview](https://vllm.ai/blog/2026-05-06-mooncake-store).
