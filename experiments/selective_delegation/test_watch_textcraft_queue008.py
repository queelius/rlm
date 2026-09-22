import importlib.util
import json
from pathlib import Path


def test_live_queue_never_triggers_analysis_and_missing_owner_is_unknown():
    assert importlib.util.find_spec("watch_textcraft_queue008") is not None
    import watch_textcraft_queue008 as w

    assert w.resolution(True, ["released"] * 4, ["released"] * 3) == "waiting"
    assert w.resolution(False, ["released", "no_owner"], ["released"] * 3) == "analyze"
    assert w.resolution(False, ["released"], ["released", "no_owner", "released"]) == "unavailable"
    assert w.resolution(False, ["owner_unresolved"], ["released"] * 3) == "waiting"


def test_actual_native_first_response_health_and_sft_has_no_inference_scope():
    assert importlib.util.find_spec("watch_textcraft_queue008") is not None
    import watch_textcraft_queue008 as w

    output = w.ROOT / "textcraft-train-readiness-001"
    path = output / "calls/t00-r0-flat-c000.json"
    call = json.loads(path.read_text())
    start = json.loads((output / "starts" / path.name).read_text())
    health = w.h.response_health(start, call)
    assert health["healthy"] and health["returned_within_90_seconds"]
    assert health["output_tokens"] == 14
    assert w.inference_scopes(Path("/not-created"), "textcraft-matched-sft-001") == []
