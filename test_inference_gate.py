import os
import tempfile

from inference_gate import inference_slot


def test_nested_job_times_out_while_the_only_slot_is_held():
    with tempfile.TemporaryDirectory() as root:
        previous_path = os.environ.get("AI_GATE_PATH")
        previous_slots = os.environ.get("AI_SHARED_SLOTS")
        os.environ["AI_GATE_PATH"] = root
        os.environ["AI_SHARED_SLOTS"] = "1"
        try:
            with inference_slot("first", timeout_seconds=0.1) as first:
                assert first
                with inference_slot("second", timeout_seconds=0.1) as second:
                    assert not second
            with inference_slot("third", timeout_seconds=0.1) as third:
                assert third
        finally:
            if previous_path is None:
                os.environ.pop("AI_GATE_PATH", None)
            else:
                os.environ["AI_GATE_PATH"] = previous_path
            if previous_slots is None:
                os.environ.pop("AI_SHARED_SLOTS", None)
            else:
                os.environ["AI_SHARED_SLOTS"] = previous_slots

