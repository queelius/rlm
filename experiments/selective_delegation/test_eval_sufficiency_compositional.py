import json

import eval_sufficiency_heldout as shared
import pytest


def test_explicit_profile_checks_component_identity_and_does_not_change_old_default(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("fixture\n")
    manifest = {
        "cases_sha256": shared.sha(cases),
        "selection_seed": 2026092205,
        "component_cluster_count": 20,
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    p = {**manifest, "manifest_sha256": shared.sha(tmp_path / "MANIFEST.json"), "metric_sha256": {}}
    assert shared.validate_panel(cases, manifest, p) == shared.sha(cases)
    with pytest.raises(ValueError, match="panel differs"):
        shared.validate_panel(cases, manifest)
    with pytest.raises(ValueError, match="component"):
        shared.validate_panel(cases, manifest, {**p, "component_cluster_count": 30})
    with pytest.raises(ValueError, match="selection"):
        shared.validate_panel(cases, manifest, {**p, "selection_seed": 2026092198})
