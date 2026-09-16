import os
import sys
import time
import signal
import logging
import argparse

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ingestion.controller import IngestionController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ControllerDaemon] %(message)s"
)
logger = logging.getLogger("ControllerDaemon")

class ControllerSupervisor:
    """
    Supervises ingestion queues, reaps crashed/expired leases,
    and monitors worker fleet health.
    """
    def __init__(self, poll_interval: float = 5.0, heartbeat_timeout: float = 60.0):
        self.poll_interval = poll_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.controller = IngestionController()
        self._running = True

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Received termination signal (%s). Shutting down Controller Supervisor...", signum)
        self._running = False

    def run_forever(self):
        logger.info(
            "Ingestion Controller Daemon started (poll_interval=%.1fs, heartbeat_timeout=%.1fs)",
            self.poll_interval, self.heartbeat_timeout
        )
        while self._running:
            try:
                # 1. Recover crashed jobs with expired worker leases
                recovered = self.controller.queue_mgr.recover_crashed_jobs()
                if recovered:
                    logger.warning("Recovered %d crashed/expired jobs into queue: %s", len(recovered), recovered)

                # 2. Check worker fleet health
                health = self.controller.get_worker_health_status(timeout_seconds=self.heartbeat_timeout)
                logger.info(
                    "Fleet Health: %d available workers (%d idle, %d busy)",
                    health["total_available_workers"],
                    health["idle_workers"],
                    health["busy_workers"]
                )

                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error("Error during controller supervisor cycle: %s", e, exc_info=True)
                time.sleep(self.poll_interval)

        logger.info("Controller Supervisor cleanly stopped.")


def main():
    parser = argparse.ArgumentParser(description="Kitchome Ingestion Controller Daemon")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Lease recovery poll interval (seconds)")
    parser.add_argument("--heartbeat-timeout", type=float, default=60.0, help="Worker dead timeout (seconds)")
    args = parser.parse_args()

    supervisor = ControllerSupervisor(
        poll_interval=args.poll_interval,
        heartbeat_timeout=args.heartbeat_timeout
    )
    supervisor.run_forever()

if __name__ == "__main__":
    main()
