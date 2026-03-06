# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time as _time

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, unpack_diffusion_output_shm
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_mq = getattr(self, "mq", None)
        if existing_mq is not None and not existing_mq.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self._lock = threading.Lock()

        # Initialize single MessageQueue for all message types (generation & RPC)
        # Assuming all readers are local for now as per current launch_engine implementation
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )

        self.result_mq = None

    def initialize_result_queue(self, handle):
        # Initialize MessageQueue for receiving results
        # We act as rank 0 reader for this queue
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized result MessageQueue")

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Sends a request to the scheduler and waits for the response."""
        with self._lock:
            try:
                # Prepare RPC request for generation
                rpc_request = {
                    "type": "rpc",
                    "method": "generate",
                    "args": (request,),
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": True,
                }

                # Broadcast RPC request to all workers
                _t_broadcast = _time.perf_counter()
                self.mq.enqueue(rpc_request)
                _t_broadcast_ms = (_time.perf_counter() - _t_broadcast) * 1000
                logger.info("Hop1 scheduler→workers: mq.enqueue (broadcast request) took %.2f ms", _t_broadcast_ms)

                # Wait for result from Rank 0 (or whoever sends it)
                if self.result_mq is None:
                    raise RuntimeError("Result queue not initialized")

                _t_dequeue = _time.perf_counter()
                output = self.result_mq.dequeue()
                _t_dequeue_ms = (_time.perf_counter() - _t_dequeue) * 1000

                _t_unpack = _time.perf_counter()
                try:
                    unpack_diffusion_output_shm(output)
                except Exception as e:
                    logger.warning("SHM unpack failed (data may already be inline): %s", e)
                _t_unpack_ms = (_time.perf_counter() - _t_unpack) * 1000

                logger.info(
                    "Hop1 scheduler←worker: mq.dequeue=%.2f ms, shm_unpack=%.2f ms (dequeue includes generation wait)",
                    _t_dequeue_ms,
                    _t_unpack_ms,
                )

                # {"status": "error", "error": str(e)}
                if isinstance(output, dict) and output.get("status") == "error":
                    raise RuntimeError("worker error")
                return output
            except zmq.error.Again:
                logger.error("Timeout waiting for response from scheduler.")
                raise TimeoutError("Scheduler did not respond in time.")

    def close(self):
        """Closes the socket and terminates the context."""
        self.mq = None
        self.result_mq = None
