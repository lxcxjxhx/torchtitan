# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise skewed KV events across real Gloo ranks without GPUs or a Store."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.generator.vllm import mooncake
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    MooncakeStoreWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


def _coordinate_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        ranks=[0, 1],
        cpu_group=dist.group.WORLD,
    )
    try:
        with patch.object(mooncake, "get_tp_group", return_value=group):
            # A follower must never query independently. Pending and hit results
            # are both shared, including the load metadata consumed by allocation.
            connector = mooncake.MooncakeStoreConnector.__new__(
                mooncake.MooncakeStoreConnector
            )
            connector.connector_worker = None
            spec = LoadSpec(16, 48, False)
            connector.connector_scheduler = SimpleNamespace(load_specs={})
            request = SimpleNamespace(request_id=f"lookup-rank{rank}")

            def lookup(*args):
                assert rank == 0
                connector.connector_scheduler.load_specs[request.request_id] = spec
                return 32, True

            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=lookup,
            ):
                assert connector.get_num_new_matched_tokens(request, 16) == (32, True)
                assert (
                    connector.connector_scheduler.load_specs[request.request_id] == spec
                )
            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                return_value=(None, False),
            ) as pending:
                assert connector.get_num_new_matched_tokens(request, 16) == (
                    None,
                    False,
                )
                assert pending.call_count == (1 if rank == 0 else 0)
            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=RuntimeError("lookup unavailable"),
            ):
                with pytest.raises(RuntimeError, match="lookup unavailable"):
                    connector.get_num_new_matched_tokens(request, 16)

            # A rank with no local work still sees another rank's pending saves.
            # The rank-0-only queue check must read cached state without a collective.
            engine = VLLMEngine.__new__(VLLMEngine)
            engine._coordinate_mooncake = True
            engine._uses_mooncake_store = True
            engine._engine = SimpleNamespace(
                output_processor=SimpleNamespace(has_unfinished_requests=lambda: False),
                engine_core=SimpleNamespace(
                    engine_core=SimpleNamespace(
                        scheduler=SimpleNamespace(has_requests=lambda: rank == 1),
                        batch_queue=None,
                    )
                ),
            )
            with patch(
                "torchtitan.rl.generator.vllm.engine.get_world_group",
                return_value=group,
            ):
                engine._refresh_mooncake_work()
            with patch.object(
                dist, "all_reduce", side_effect=AssertionError("unexpected collective")
            ):
                assert engine.has_pending_work
                assert not engine.has_unfinished_requests()

            # Use upstream block accounting: the first save completion must not
            # release references, and a failed receive must wait for the other
            # worker before the scheduler can decide to retry/recompute.
            pool = BlockPool(8, False, 16)
            blocks = pool.get_new_blocks(2)
            store_scheduler = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
            store_scheduler._gpu_block_pool = pool
            store_scheduler._pinned_saves = {7: ([b.block_id for b in blocks], 2)}
            scheduler = mooncake.MooncakeScheduler.__new__(mooncake.MooncakeScheduler)
            scheduler._kv_output_aggregator = KVOutputAggregator(2)
            local_ids = {
                logical: f"{logical}-rank{rank}"
                for logical in ("load", "failed", "send")
            }
            scheduler.requests = dict.fromkeys(local_ids.values())
            scheduler._logical_request_ids = {
                local: logical for logical, local in local_ids.items()
            }
            scheduler._local_request_ids = local_ids.copy()
            observed = []

            def apply_output(_self, _scheduled, output):
                kv = output.kv_connector_output
                store_scheduler.update_connector_output(kv)
                observed.append(kv)
                # Sampling output stays local; only control metadata is gathered.
                assert output.sampled_token_ids == [[rank]]
                assert output.req_ids == [local_ids["load"]]
                return {}

            with patch.object(
                mooncake.Scheduler, "update_from_output", new=apply_output
            ):
                for step in range(3):
                    report = rank == 0 and step == 0 or rank == 1 and step == 2
                    kv = KVConnectorOutput(
                        finished_recving=(
                            {local_ids["load"], local_ids["failed"]} if report else None
                        ),
                        finished_sending={local_ids["send"]} if report else None,
                        failed_recving=(
                            {local_ids["failed"]} if rank == 0 and step == 0 else set()
                        ),
                        kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                            {7: 1} if report else {}
                        ),
                    )
                    output = ModelRunnerOutput(
                        req_ids=[local_ids["load"]],
                        req_id_to_index={local_ids["load"]: 0},
                        sampled_token_ids=[[rank]],
                        kv_connector_output=kv,
                    )
                    scheduler.update_from_output(None, output)
                    if step < 2:
                        assert not observed[-1].finished_recving
                        assert not observed[-1].finished_sending
                        assert not observed[-1].failed_recving
                        assert all(b.ref_cnt == 1 for b in blocks)
                        assert store_scheduler.has_pending_push_work()
                    else:
                        assert observed[-1].finished_recving == {
                            local_ids["load"],
                            local_ids["failed"],
                        }
                        assert observed[-1].finished_sending == {local_ids["send"]}
                        assert observed[-1].failed_recving == {local_ids["failed"]}
                        assert all(b.ref_cnt == 0 for b in blocks)
                        assert not store_scheduler.has_pending_push_work()
    finally:
        dist.destroy_process_group()


def test_lookup_and_delayed_completion_coordination(tmp_path):
    mp.spawn(_coordinate_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)


def test_idle_dp_forward_does_not_skip_queued_output_drain():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._coordinate_mooncake = True
    engine._session_kv_holder = None
    calls = []
    engine._refresh_mooncake_work = lambda: calls.append("refresh")

    def coordinate(has_requests):
        assert not has_requests
        engine._engine.should_execute_dummy_batch = True
        calls.append("coordinate")

    def drain():
        assert not engine._engine.should_execute_dummy_batch
        calls.append("drain")
        return []

    engine._engine = SimpleNamespace(
        dp_group=object(),
        has_unfinished_requests_dp=coordinate,
        engine_core=SimpleNamespace(
            engine_core=SimpleNamespace(
                scheduler=SimpleNamespace(has_requests=lambda: False),
                batch_queue=[object()],
            ),
            execute_dummy_batch=lambda: calls.append("dummy"),
        ),
        step=drain,
    )
    assert engine.step() == []
    assert calls == ["coordinate", "dummy", "drain", "refresh"]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_uses_resolved_async_setting(async_scheduling):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
    )
    with (
        patch.object(mooncake, "MooncakeScheduler") as sync,
        patch.object(mooncake, "AsyncMooncakeScheduler") as async_,
    ):
        result = mooncake.create_mooncake_scheduler(config)
        selected, other = (async_, sync) if async_scheduling else (sync, async_)
        assert result is selected.return_value
        selected.assert_called_once_with(config)
        other.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_each_dp_replica_has_own_lookup_server_and_save_quorum(rank):
    worker = SimpleNamespace(lookup_server=None)
    scheduler = SimpleNamespace(_num_workers=4)

    def init(connector, *args):
        connector.connector_worker = worker
        connector.connector_scheduler = scheduler

    group = SimpleNamespace(rank_in_group=rank, world_size=2)
    with (
        patch.object(mooncake.VllmMooncakeStoreConnector, "__init__", new=init),
        patch.object(mooncake, "get_tp_group", return_value=group),
        patch.object(mooncake, "LookupKeyServer") as server,
    ):
        connector = mooncake.MooncakeStoreConnector(Mock(), Mock())
        assert connector.connector_scheduler._num_workers == 2
        assert server.call_count == (1 if rank == 0 else 0)
        # Prevent the upstream destructor from calling close on this fake worker.
        connector.connector_worker = None


@pytest.mark.parametrize("suffix", ["00000000", "ffffffff"])
def test_priority_ties_use_logical_ids_across_waiting_queues(suffix):
    # Native vLLM orders these IDs differently depending on the random suffix.
    # A TP leader's lookup must never be applied to another request on a follower.
    scheduler = mooncake.MooncakeScheduler.__new__(mooncake.MooncakeScheduler)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler._logical_request_ids = {}
    scheduler._local_request_ids = {}
    scheduler.requests = {}
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler.spec_decode_metrics_level = "none"
    scheduler.waiting = mooncake._LogicalPriorityRequestQueue(
        scheduler._logical_request_ids
    )
    scheduler.skipped_waiting = mooncake._LogicalPriorityRequestQueue(
        scheduler._logical_request_ids
    )
    requests = []
    for logical_id in ("request-5", "request"):
        request = Request(
            f"{logical_id}-{suffix}",
            [1],
            SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": {"titan_request_id": logical_id}},
            ),
            None,
            arrival_time=1.0,
        )
        scheduler.add_request(request)
        requests.append(request)
    earlier = requests[1]
    assert scheduler.waiting.pop_request() is earlier
    # The same tie-break must hold between the ready and blocked queue heads.
    earlier.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(earlier)
    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.skipped_waiting
    scheduler.waiting.prepend_requests(scheduler.skipped_waiting)
    scheduler.skipped_waiting.remove_request(earlier)
    assert list(scheduler.waiting) == [earlier, requests[0]]
    scheduler.waiting.remove_requests([earlier])
    scheduler.waiting.prepend_request(earlier)
    assert scheduler.waiting.pop_request() is earlier
    assert scheduler.waiting.pop_request() is requests[0]
    assert not scheduler.waiting


@pytest.mark.parametrize("late_receive", [False, True])
@pytest.mark.parametrize(
    "finished_status", [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR]
)
def test_terminal_cleanup_preserves_block_identity(late_receive, finished_status):
    # Use real queues and KV allocation/freeing without constructing a GPU worker.
    # Per-process string hash order must not change a later invalid block's owner.
    scheduler = mooncake.MooncakeScheduler.__new__(mooncake.MooncakeScheduler)
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.defer_block_free = False
    scheduler._inflight_prefills = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.requests = {}
    scheduler._logical_request_ids = {}
    scheduler._local_request_ids = {}
    scheduler.spec_decode_metrics_level = "none"
    scheduler.log_stats = False
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.encoder_cache_manager = EncoderCacheManager(0)
    scheduler.block_size = 16
    manager = scheduler.kv_cache_manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=5,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
                    ),
                )
            ],
        ),
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )

    def allocate(request_id, logical_id):
        request = Request(
            request_id,
            [1] * 32,
            SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": {"titan_request_id": logical_id}},
            ),
            None,
        )
        scheduler.add_request(request)
        assert manager.allocate_slots(request, num_new_tokens=32) is not None
        request.num_computed_tokens = 32
        return request

    # Deliberately reverse local-ID ordering: block release must use logical IDs.
    for request_id, logical_id in (("local-B", "request-A"), ("local-A", "request-B")):
        request = allocate(request_id, logical_id)
        scheduler.waiting.remove_request(request)
        if late_receive:
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            scheduler.skipped_waiting.add_request(request)
        else:
            request.status = RequestStatus.RUNNING
            scheduler.running.append(request)
    original_b_blocks = manager.get_block_ids("local-A")[0].copy()
    scheduler.finish_requests({"local-A", "local-B", "unknown"}, finished_status)
    if late_receive:
        assert manager.block_pool.get_num_free_blocks() == 0
        assert len(scheduler._local_request_ids) == 2
        scheduler._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={"local-A", "local-B"})
        )
    assert not scheduler.requests
    assert not scheduler._logical_request_ids and not scheduler._local_request_ids
    assert (
        not scheduler.running
        and not scheduler.waiting
        and not scheduler.skipped_waiting
    )

    # A logical ID can be reused after its old request/receive is fully retired.
    following = [allocate("next-C", "request-A"), allocate("next-D", "request-B")]
    # Non-cached blocks are reused LIFO: canonical A-then-B cleanup puts B first.
    assert manager.get_block_ids("next-C")[0] == original_b_blocks[::-1]
    affected, _, _ = scheduler._update_requests_with_invalid_blocks(
        following, {original_b_blocks[-1]}, {}, evict_blocks=False
    )
    assert affected == {"next-C"}
