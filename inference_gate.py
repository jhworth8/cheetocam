"""Cross-process scheduling for heavyweight local inference.

Every camera has its own capture/detection process so a broken wireless feed
cannot stop the front-door webcam. Florence and CLIP are different: allowing
every camera to saturate all CPU cores at once makes every feed worse. This
small file-lock semaphore gives those heavyweight jobs a shared number of
slots without putting camera credentials or frames into another service.
"""

from contextlib import contextmanager
import fcntl
import logging
import os
import time


def _slot_count():
    try:
        return max(1, min(8, int(os.getenv("AI_SHARED_SLOTS", "1"))))
    except (TypeError, ValueError):
        return 1


@contextmanager
def inference_slot(label, timeout_seconds=None):
    """Yield True after claiming a shared inference slot, or False on timeout.

    Locks are automatically released by the kernel if a detector crashes. The
    caller chooses whether timeout means "continue without a caption" or a
    retry; camera capture never depends on a permanently healthy coordinator.
    """
    root = os.getenv("AI_GATE_PATH", "/tmp/critter-cam-ai").strip()
    if not root:
        root = "/tmp/critter-cam-ai"
    deadline = None if timeout_seconds is None else time.monotonic() + max(
        0.0, float(timeout_seconds)
    )
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
    except OSError as exc:
        logging.warning("Local-AI gate unavailable (%s); continuing ungated.", exc)
        yield True
        return

    waited_at = time.monotonic()
    while True:
        for index in range(_slot_count()):
            handle = None
            try:
                handle = open(os.path.join(root, f"slot-{index}.lock"), "a+")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                waited = time.monotonic() - waited_at
                if waited >= 0.5:
                    logging.info(
                        "Local-AI job %s waited %.1fs for shared slot %d.",
                        label,
                        waited,
                        index + 1,
                    )
                try:
                    yield True
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                return
            except BlockingIOError:
                if handle is not None:
                    handle.close()
            except OSError as exc:
                if handle is not None:
                    handle.close()
                logging.warning(
                    "Local-AI gate failed for %s (%s); continuing ungated.",
                    label,
                    exc,
                )
                yield True
                return

        if deadline is not None and time.monotonic() >= deadline:
            logging.warning("Local-AI job %s timed out waiting for a shared slot.", label)
            yield False
            return
        time.sleep(0.1)

