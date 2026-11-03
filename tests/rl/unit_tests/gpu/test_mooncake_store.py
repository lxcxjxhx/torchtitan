# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two independent GPU generators sharing a real Mooncake CPU store.

Run explicitly with RUN_MOONCAKE_TEST=1. Requires mooncake-transfer-engine,
mooncake_master on PATH, and two idle GPUs with working GPUDirect RDMA.
"""

import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import time
import traceback
import uuid

import pytest
import torch


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _generator_worker(
    device_id, port, connector_args, hold_sessions, block_size, commands, results
):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=device_id,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
    )
    try:
        import torch.distributed as dist

        from torchtitan.config import DebugConfig
        from torchtitan.models.qwen3 import build_model_config
        from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
        from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
        from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
        from torchtitan.rl.generator.vllm.engine import VLLMEngine
        from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
        from torchtitan.tools.utils import has_cuda_capability
        from vllm.config import KVTransferConfig
        from vllm.distributed import cleanup_dist_env_and_memory

        torch.cuda.set_device(0)
        dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
        parallelism = InferenceParallelismConfig()
        model = build_model_config(
            "debugmodel",
            parallelism=parallelism.to_training(),
            seq_len=2048,
        )
        model.local_compile.regions = []
        engine = VLLMEngine.Config(
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            kv_transfer_config=KVTransferConfig(**connector_args),
            session_kv_holder=(VLLMSessionKVHolder.Config() if hold_sessions else None),
            # Pre-SM90 engines already select 256-token blocks. Exercise that
            # granularity on newer GPUs too, without overriding an owned arg.
            extra_engine_args=(
                {"block_size": block_size}
                if block_size is not None and has_cuda_capability(9, 0)
                else {}
            ),
        ).build(
            model=model,
            parallelism=parallelism,
            model_path="tests/assets/tokenizer",
            max_num_seqs=4,
            model_dtype="bfloat16",
            gpu_memory_limit=0.15,
            max_num_batched_tokens=1024,
            checkpointer=None,
            dist_moe_runtime=None,
            debug=DebugConfig(seed=42),
            reset_kv_cache_on_model_state_dict_load=False,
            rank=0,
            generator_name=f"generator-{device_id}",
            output_dir="/tmp",
        )
        # Use real model weights in the normal pinned receive buffers.
        torch.manual_seed(42)
        for tensor in engine.prefetched_model_state_dict.values():
            if tensor.ndim == 1:
                tensor.fill_(1.0)
            else:
                tensor.normal_(mean=0.0, std=0.02)
        engine.load_prefetched_model_state_dict(0)
        block_size = engine._engine.engine_core.engine_core.scheduler.block_size
        results.put(
            ("ready", (block_size, str(torch.cuda.get_device_properties(0).uuid)))
        )
        while (command := commands.get()) is not None:
            version, prompt, clear_local = command
            if version != engine._loaded_policy_version:
                # Change real weights, not just the version label.
                for tensor in engine.prefetched_model_state_dict.values():
                    tensor.mul_(0.75)
                engine.load_prefetched_model_state_dict(version)
            if clear_local:
                engine.release_sessions(["session-0"], [])
                assert engine._engine.reset_prefix_cache(reset_connector=False)
            request = EngineRequest(
                request_id=uuid.uuid4().hex,
                prompt_token_ids=prompt,
                # Leave room for the uncomputed final token and async scheduling
                # lag beyond a complete block, so reuse proves decode offloading.
                sampling=SamplingConfig(
                    temperature=0.0, max_tokens=max(48, block_size + 2), seed=42
                ),
                group_id=0,
                routing_session_id="session-0",
                arrival_time=time.time(),
            )
            # Keep the group pinned to 0 even after updating weights.
            request.min_policy_version = 0
            assert engine.add_requests([request]) == []
            finished = []
            deadline = time.monotonic() + 60
            # Keep stepping background transfers through the same engine contract
            # used by EngineLoop, including after the last request output finishes.
            while engine.has_pending_work or engine.has_unfinished_requests():
                assert time.monotonic() < deadline, "KV transfer failed to drain"
                finished.extend(engine.step())
            assert len(finished) == 1
            if hold_sessions:
                assert engine._session_kv_holder.num_sessions == 1
                held = engine._session_kv_holder._held["session-0"]
                assert held and all(block.ref_cnt > 0 for block in held)
            output = finished[0]
            results.put(
                (
                    "output",
                    (
                        output.token_ids,
                        output.token_logprobs,
                        output.metrics_inputs.num_cached_tokens,
                    ),
                )
            )
        engine.shutdown()
        engine._engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()
    except BaseException:
        results.put(("error", traceback.format_exc()))
        raise


@pytest.mark.skipif(
    os.getenv("RUN_MOONCAKE_TEST") != "1",
    reason="opt-in Mooncake RDMA integration test",
)
@pytest.mark.parametrize(
    "roles, save_decode_cache, hold_sessions, block_size",
    [
        pytest.param(("kv_both", "kv_both"), True, False, 256, id="decode"),
        pytest.param(("kv_both", "kv_both"), False, False, None, id="prompt-only"),
        pytest.param(("kv_producer", "kv_consumer"), None, False, None, id="roles"),
        pytest.param(("kv_both", "kv_both"), True, True, None, id="session-holder"),
    ],
)
def test_shared_cpu_kv_and_weight_version_isolation(
    tmp_path, monkeypatch, roles, save_decode_cache, hold_sessions, block_size
):
    assert torch.cuda.device_count() >= 2
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = (
        [device.strip() for device in visible_devices.split(",")]
        if visible_devices is not None
        else ["0", "1"]
    )
    master = shutil.which("mooncake_master")
    assert master is not None, "install mooncake-transfer-engine"
    port = _free_port()
    config = tmp_path / "store.json"
    config.write_text(
        json.dumps(
            {
                "mode": "embedded",
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": f"127.0.0.1:{port}",
                "global_segment_size": "1GB",
                "local_buffer_size": "64MB",
                "protocol": "rdma",
                "device_name": "",
                "enable_offload": False,
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config))
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    context = multiprocessing.get_context("spawn")
    namespace = f"titan-test-{uuid.uuid4().hex}"
    processes = []
    with (tmp_path / "master.log").open("w") as log, subprocess.Popen(
        [master, f"--rpc_port={port}", "--rpc_thread_num=4"],
        stdout=log,
        stderr=log,
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    assert server.poll() is None
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            queues = []
            block_sizes = []
            for gpu in range(2):
                extra = {"cache_prefix": namespace}
                if save_decode_cache is not None:
                    extra["save_decode_cache"] = save_decode_cache
                connector_args = {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": roles[gpu],
                    "kv_connector_extra_config": extra,
                }
                commands, results = context.Queue(), context.Queue()
                process = context.Process(
                    target=_generator_worker,
                    args=(
                        device_ids[gpu],
                        _free_port(),
                        connector_args,
                        hold_sessions,
                        block_size,
                        commands,
                        results,
                    ),
                )
                process.start()
                processes.append(process)
                queues.append((commands, results))
                kind, payload = results.get(timeout=300)
                assert kind == "ready", payload
                resolved_block_size, device_uuid = payload
                assert device_uuid == str(torch.cuda.get_device_properties(gpu).uuid)
                if block_size is not None:
                    assert resolved_block_size == block_size
                block_sizes.append(resolved_block_size)

            assert block_sizes[0] == block_sizes[1]
            cache_block_size = block_sizes[0]

            def generate(gpu, version, prompt, clear_local=False):
                commands, results = queues[gpu]
                commands.put((version, prompt, clear_local))
                kind, payload = results.get(timeout=90)
                assert kind == "output", payload
                return payload

            prompt = list(range(100, 100 + max(512, 2 * cache_block_size)))
            # Recomputing the last prompt token limits a repeated prompt's hit
            # to the preceding full block.
            cached_prompt_tokens = (
                (len(prompt) - 1) // cache_block_size * cache_block_size
            )
            first = generate(0, 0, prompt)
            assert first[2] == 0

            # B has never seen this prefix: its first hit must come from A's store.
            next_turn = prompt + first[0] + [42, 43]
            continued = generate(1, 0, next_turn)
            if save_decode_cache:
                assert continued[2] >= len(prompt) + cache_block_size
            else:
                assert cached_prompt_tokens <= continued[2] <= len(prompt)
            shared = generate(1, 0, prompt, clear_local=True)
            assert shared[2] >= cached_prompt_tokens
            assert shared[0] == first[0]
            torch.testing.assert_close(
                torch.tensor(shared[1]), torch.tensor(first[1]), atol=0.02, rtol=0
            )

            updated = generate(1, 1, prompt)
            assert updated[2] == 0, "old policy KV must not match after a weight update"
            assert updated[1] != first[1], "the test must actually change model weights"
            # Updating B must not purge A's still-valid version from the shared store.
            old = generate(0, 0, prompt, clear_local=True)
            assert old[2] >= cached_prompt_tokens
            assert old[0] == first[0]
            print(
                f"Mooncake cached tokens: cold={first[2]}, shared={shared[2]}, "
                f"next_turn={continued[2]}, new_policy={updated[2]}, old_policy={old[2]}"
            )
            for commands, _ in queues:
                commands.put(None)
            for process in processes:
                process.join(timeout=30)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
            server.terminate()
            server.wait(timeout=10)
