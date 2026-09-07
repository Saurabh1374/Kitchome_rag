import os
import sys
import time
import signal
import logging
import argparse
import multiprocessing
import threading
from typing import List, Optional, Any, Union

from config import config
from src.ingestion.worker import ChunkerWorker, EmbedderWorker, IngestionWorker
from src.ingestion.embedder import get_embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(processName)s/%(threadName)s] %(message)s"
)
logger = logging.getLogger("WorkerRunner")


def _run_chunker_target(
    worker_id: str,
    poll_interval: float = 1.0,
    max_iterations: Optional[int] = None,
    stop_event: Optional[Any] = None
):
    """Target function for chunker worker instances."""
    logger.info(f"Starting ChunkerWorker instance: {worker_id}")
    worker = ChunkerWorker(worker_id=worker_id)
    worker.run_forever(
        poll_interval=poll_interval,
        max_iterations=max_iterations,
        stop_event=stop_event
    )


def _run_embedder_target(
    worker_id: str,
    batch_size: int = 32,
    poll_interval: float = 1.0,
    max_iterations: Optional[int] = None,
    stop_event: Optional[Any] = None,
    embedder_overrides: Optional[dict] = None
):
    """Target function for embedder swarm worker instances."""
    logger.info(f"Starting EmbedderWorker instance: {worker_id} (batch_size={batch_size})")
    embedder = get_embedder(**embedder_overrides) if embedder_overrides else None
    worker = EmbedderWorker(worker_id=worker_id, batch_size=batch_size, embedder=embedder)
    worker.run_forever(
        poll_interval=poll_interval,
        max_iterations=max_iterations,
        stop_event=stop_event
    )


def _run_unified_target(
    worker_id: str,
    poll_interval: float = 1.0,
    max_iterations: Optional[int] = None,
    stop_event: Optional[Any] = None,
    embedder_overrides: Optional[dict] = None
):
    """Target function for unified ingestion worker instances."""
    logger.info(f"Starting IngestionWorker instance: {worker_id}")
    embedder = get_embedder(**embedder_overrides) if embedder_overrides else None
    worker = IngestionWorker(worker_id=worker_id, embedder=embedder)
    worker.run_forever(
        poll_interval=poll_interval,
        max_iterations=max_iterations,
        stop_event=stop_event
    )


class WorkerSupervisor:
    """
    Supervisor that manages worker lifecycle, concurrency, and graceful shutdown.
    Supports both multi-process (bypasses GIL) and multi-threaded execution.
    """
    def __init__(
        self,
        role: str = "all",
        mode: str = "process",
        chunker_concurrency: int = 1,
        embedder_concurrency: int = 2,
        batch_size: int = 32,
        poll_interval: float = 1.0,
        max_iterations: Optional[int] = None,
        embedder_overrides: Optional[dict] = None
    ):
        self.role = role
        self.mode = mode
        self.chunker_concurrency = chunker_concurrency
        self.embedder_concurrency = embedder_concurrency
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_iterations = max_iterations
        self.embedder_overrides = embedder_overrides or {}

        if self.mode == "process":
            self.stop_event = multiprocessing.Event()
        else:
            self.stop_event = threading.Event()

        self.workers: List[Union[multiprocessing.Process, threading.Thread]] = []

    def start(self):
        """Spawns worker instances based on configured role and concurrency."""
        worker_cls = multiprocessing.Process if self.mode == "process" else threading.Thread

        if self.role in ("all", "chunker"):
            for i in range(self.chunker_concurrency):
                w_id = f"chunker_{self.mode}_{i+1}"
                t = worker_cls(
                    target=_run_chunker_target,
                    args=(w_id, self.poll_interval, self.max_iterations, self.stop_event),
                    name=w_id
                )
                self.workers.append(t)
                t.start()

        if self.role in ("all", "embedder"):
            for i in range(self.embedder_concurrency):
                w_id = f"embedder_{self.mode}_{i+1}"
                t = worker_cls(
                    target=_run_embedder_target,
                    args=(w_id, self.batch_size, self.poll_interval, self.max_iterations, self.stop_event, self.embedder_overrides),
                    name=w_id
                )
                self.workers.append(t)
                t.start()

        if self.role == "unified":
            w_id = f"unified_{self.mode}_1"
            t = worker_cls(
                target=_run_unified_target,
                args=(w_id, self.poll_interval, self.max_iterations, self.stop_event, self.embedder_overrides),
                name=w_id
            )
            self.workers.append(t)
            t.start()

        logger.info(
            f"WorkerSupervisor started {len(self.workers)} workers "
            f"(mode={self.mode}, role={self.role})"
        )

    def stop(self, timeout: float = 10.0):
        """Signals all workers to stop and waits for clean exit."""
        logger.info("WorkerSupervisor initiating shutdown...")
        self.stop_event.set()

        for w in self.workers:
            w.join(timeout=timeout)
            if hasattr(w, "is_alive") and w.is_alive():
                logger.warning(f"Worker {w.name} did not stop cleanly within {timeout}s.")
                if hasattr(w, "terminate"):
                    logger.info(f"Terminating worker process {w.name}")
                    w.terminate()

        logger.info("WorkerSupervisor shutdown complete.")

    def run_until_interrupted(self):
        """Monitors workers until SIGINT/SIGTERM or until all workers terminate."""
        def _handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown.")
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError):
            # Signal handling might not be available in non-main threads
            pass

        self.start()
        try:
            while not self.stop_event.is_set():
                alive_count = sum(1 for w in self.workers if w.is_alive())
                if alive_count == 0:
                    logger.info("All workers have completed their assigned tasks.")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected in supervisor main thread.")
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Kitchome Ingestion Worker Daemon Runner")
    parser.add_argument(
        "--role",
        choices=["all", "chunker", "embedder", "unified"],
        default=config.worker.role,
        help="Worker pool role to run"
    )
    parser.add_argument(
        "--mode",
        choices=["process", "thread"],
        default=config.worker.mode,
        help="Concurrency mode: 'process' (bypasses GIL) or 'thread' (low-RAM)"
    )
    parser.add_argument(
        "--chunker-concurrency",
        type=int,
        default=config.worker.chunker_concurrency,
        help="Number of concurrent chunker instances"
    )
    parser.add_argument(
        "--embedder-concurrency",
        type=int,
        default=config.worker.embedder_concurrency,
        help="Number of concurrent embedder instances"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.worker.embedder_batch_size,
        help="Number of chunks claimed per embedding batch"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=config.worker.poll_interval_seconds,
        help="Seconds to sleep when idle"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional maximum loop iterations (useful for testing or one-off batch runs)"
    )
    parser.add_argument(
        "--embedder-provider",
        choices=["hash", "huggingface", "huggingface_api", "ollama", "openai", "fastembed", "mock"],
        default=None,
        help="Embedder provider override (e.g. hash, huggingface, huggingface_api)"
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help="Embedder model name override"
    )
    parser.add_argument(
        "--embedder-dimension",
        type=int,
        default=None,
        help="Embedder vector dimension override"
    )
    parser.add_argument(
        "--embedder-device",
        type=str,
        default=None,
        help="Embedder device override (e.g. cpu, mps, cuda)"
    )

    args = parser.parse_args()

    embedder_overrides = {}
    if args.embedder_provider:
        embedder_overrides["provider"] = args.embedder_provider
    if args.embedder_model:
        embedder_overrides["model_name"] = args.embedder_model
    if args.embedder_dimension:
        embedder_overrides["dimension"] = args.embedder_dimension
    if args.embedder_device:
        embedder_overrides["device"] = args.embedder_device

    supervisor = WorkerSupervisor(
        role=args.role,
        mode=args.mode,
        chunker_concurrency=args.chunker_concurrency,
        embedder_concurrency=args.embedder_concurrency,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        max_iterations=args.max_iterations,
        embedder_overrides=embedder_overrides
    )
    supervisor.run_until_interrupted()


if __name__ == "__main__":
    main()
