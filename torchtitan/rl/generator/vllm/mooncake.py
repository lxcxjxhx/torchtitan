# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coordinate Mooncake cache decisions between external-launcher TP schedulers.

Each DP replica has its own requests and TP group. Only KV control metadata is
exchanged; the upstream connector still transfers each worker's KV shard.
Titan's logical request IDs coordinate ranks; vLLM's internal IDs stay local.
TODO(acisseJZhong): upstream external-launcher KV coordination so these adapters
can be removed: https://github.com/vllm-project/vllm/issues/11400
"""

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector as VllmMooncakeStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    LookupKeyServer,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


class MooncakeStoreConnector(VllmMooncakeStoreConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        group = get_tp_group()
        if self.connector_scheduler is not None:
            # external_launcher world_size includes DP, whose requests and save
            # jobs are independent. A save waits only for this TP group's shards.
            self.connector_scheduler._num_workers = group.world_size
        worker = self.connector_worker
        if (
            worker is not None
            and group.rank_in_group == 0
            and worker.lookup_server is None
        ):
            # Upstream starts this only on global rank 0; each DP replica needs
            # its own server at the connector's DP-local IPC address.
            worker.lookup_server = LookupKeyServer(worker, vllm_config)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        group = get_tp_group()
        if group.world_size == 1:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)

        # The lookup service checks every required TP shard. Publish its whole
        # result, including pending status and partial-hit load metadata, once.
        payload: list[Any] = [None]
        if group.rank_in_group == 0:
            try:
                result = super().get_num_new_matched_tokens(
                    request, num_computed_tokens
                )
                payload[0] = (
                    result,
                    self.connector_scheduler.load_specs.get(request.request_id),
                    None,
                )
            except Exception as exc:
                payload[0] = (None, None, str(exc))
        dist.broadcast_object_list(payload, src=group.ranks[0], group=group.cpu_group)
        result, load_spec, error = payload[0]
        if error is not None:
            raise RuntimeError(f"Mooncake lookup failed on TP rank 0: {error}")
        if load_spec is not None:
            self.connector_scheduler.load_specs[request.request_id] = load_spec
        return result


class _LogicalPriorityRequestQueue(RequestQueue):
    """A priority heap with a shared logical-ID tie-break instead of local IDs."""

    def __init__(self, logical_ids: dict[str, str]) -> None:
        self._logical_ids = logical_ids
        self._heap: list[tuple[int, float, str, Request]] = []

    def _entry(self, request: Request) -> tuple[int, float, str, Request]:
        return (
            request.priority,
            request.arrival_time,
            self._logical_ids[request.request_id],
            request,
        )

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, self._entry(request))

    def pop_request(self) -> Request:
        return heapq.heappop(self._heap)[-1]

    def peek_request(self) -> Request:
        return self._heap[0][-1]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap.remove(self._entry(request))
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        ids = {request.request_id for request in requests}
        self._heap = [entry for entry in self._heap if entry[-1].request_id not in ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        return (entry[-1] for entry in sorted(self._heap))


class MooncakeScheduler(Scheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._kv_output_aggregator = KVOutputAggregator.from_connector(
            self.connector, get_tp_group().world_size
        )
        self._logical_request_ids: dict[str, str] = {}
        self._local_request_ids: dict[str, str] = {}
        if self.policy == SchedulingPolicy.PRIORITY:
            self.waiting = _LogicalPriorityRequestQueue(self._logical_request_ids)
            self.skipped_waiting = _LogicalPriorityRequestQueue(
                self._logical_request_ids
            )

    def add_request(self, request: Request) -> None:
        # Native KV transfer metadata preserves Titan's ID through input
        # processing without changing vLLM's randomized internal request ID.
        assert request.kv_transfer_params is not None
        logical_id = request.kv_transfer_params["titan_request_id"]
        local_id = request.request_id
        assert local_id not in self._logical_request_ids
        assert logical_id not in self._local_request_ids
        self._logical_request_ids[local_id] = logical_id
        self._local_request_ids[logical_id] = local_id
        super().add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if (
            self.policy == SchedulingPolicy.PRIORITY
            and self.waiting
            and self.skipped_waiting
        ):

            def key(queue: RequestQueue) -> tuple[int, float, str]:
                request = queue.peek_request()
                return (
                    request.priority,
                    request.arrival_time,
                    self._logical_request_ids[request.request_id],
                )

            return min((self.waiting, self.skipped_waiting), key=key)
        return super()._select_waiting_queue_for_scheduling()

    def _free_blocks(self, request: Request) -> None:
        super()._free_blocks(request)
        # Keep IDs while a cancelled receive still owns the request. Store saves
        # own separate integer job IDs and block references, not request IDs.
        logical_id = self._logical_request_ids.pop(request.request_id)
        del self._local_request_ids[logical_id]

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        group = get_tp_group()
        if group.world_size > 1:
            outputs: list[KVConnectorOutput | None] = [None] * group.world_size
            dist.all_gather_object(
                outputs,
                _remap_request_ids(
                    model_runner_output.kv_connector_output, self._logical_request_ids
                ),
                group=group.cpu_group,
            )
            # Reuse upstream aggregation of delayed receive completions, errors
            # and completed_saves. Model outputs and GPU tensors remain local.
            aggregated_output = self._kv_output_aggregator.aggregate(
                [replace(model_runner_output, kv_connector_output=o) for o in outputs]
            )
            assert aggregated_output is not None
            aggregated_output.kv_connector_output = _remap_request_ids(
                aggregated_output.kv_connector_output, self._local_request_ids
            )
            model_runner_output = aggregated_output
        return super().update_from_output(scheduler_output, model_runner_output)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        # Free in the same order on every TP rank so future physical block IDs
        # agree. Upstream converts iterable IDs back to a set; pass strings.
        if isinstance(request_ids, str):
            return super().finish_requests(request_ids, finished_status)
        ids = self.requests if request_ids is None else request_ids
        finished = []
        for request_id in sorted(
            set(ids) & self.requests.keys(), key=self._logical_request_ids.__getitem__
        ):
            finished.extend(super().finish_requests(request_id, finished_status))
        return finished

    def _update_from_kv_xfer_finished(self, output: KVConnectorOutput) -> None:
        # A cancelled/failed load may retain blocks until its receive finishes.
        # These completion sets also need deterministic release order; otherwise
        # an aggregated invalid_block_id can refer to different requests by rank.
        if self.connector is not None:
            self.connector.update_connector_output(output)
        for request_id in sorted(
            output.finished_recving or (), key=self._logical_request_ids.__getitem__
        ):
            request = self.requests[request_id]
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(request_id)
            else:
                assert RequestStatus.is_finished(request.status)
                self._free_blocks(request)
        for request_id in sorted(
            output.finished_sending or (), key=self._logical_request_ids.__getitem__
        ):
            self._free_blocks(self.requests[request_id])


def _remap_request_ids(
    output: KVConnectorOutput | None, request_ids: dict[str, str]
) -> KVConnectorOutput | None:
    if output is None:
        return None
    return replace(
        output,
        finished_recving=(
            {request_ids[req_id] for req_id in output.finished_recving}
            if output.finished_recving is not None
            else None
        ),
        finished_sending=(
            {request_ids[req_id] for req_id in output.finished_sending}
            if output.finished_sending is not None
            else None
        ),
        failed_recving={request_ids[req_id] for req_id in output.failed_recving},
    )


class AsyncMooncakeScheduler(AsyncScheduler, MooncakeScheduler):
    """Preserve vLLM's async scheduling while coordinating KV completions."""


def create_mooncake_scheduler(
    vllm_config: VllmConfig,
    *args: Any,
    **kwargs: Any,
) -> Scheduler:
    # vLLM resolves automatic async scheduling while building VllmConfig, after
    # Titan creates EngineArgs. Select the matching implementation here.
    scheduler_cls = (
        AsyncMooncakeScheduler
        if vllm_config.scheduler_config.async_scheduling
        else MooncakeScheduler
    )
    return scheduler_cls(vllm_config, *args, **kwargs)
