"""Fork an accepted capture checkpoint without rewriting failed evidence.

Offline file operation: the caller must pause the source run first. No camera,
robot, SDK or ROS access. The new attempt must qualify and validate a new fit.
"""

import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .handeye import POLICY


def load_prior_validation(data, run_path, _seen=None):
    """Read and verify archived failures for an unchanged physical setup.

    The bundle is separate from active capture records. Its observations may
    only be evaluated, never fitted. Each source run and image is hash-checked;
    nested restarts cannot shed an earlier failed validation view.
    """
    recovery = data.get("recovery")
    if not recovery:
        return None
    unchanged = recovery.get("mount_unchanged")
    if unchanged is None:
        # This exact confirmation was written by the first recovery version.
        unchanged = "board and mount confirmed unchanged" in recovery.get("reason", "").lower()
    if unchanged is not True:
        raise ValueError(
            "Changed or unconfirmed mounting requires a fresh run; archived poses cannot be mixed."
        )
    run_path = Path(run_path).resolve()
    parent = Path(recovery["parent_run_path"]).resolve()
    if parent.parent != run_path.parent or parent.name != recovery["parent_run_id"]:
        raise ValueError("Archived validation parent is not the recorded sibling run.")
    seen = set() if _seen is None else set(_seen)
    if parent in seen or parent == run_path:
        raise ValueError("Archived validation lineage contains a cycle.")
    seen.add(run_path)
    raw = (parent / "run.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != recovery["parent_run_sha256"]:
        raise ValueError("Archived validation run no longer matches its recorded hash.")
    original = json.loads(raw)
    for key in ("board", "calibration_sha256", "urdf_sha256", "side", "base_frame", "tool_frame"):
        if data.get(key) != original.get(key):
            raise ValueError(
                f"Archived validation uses different {key}; start a fresh physical setup."
            )
    inherited = load_prior_validation(original, parent, seen)
    samples = copy.deepcopy(inherited["samples"]) if inherited else []
    sources = copy.deepcopy(inherited["sources"]) if inherited else []
    failures = {
        row["id"] for row in (original.get("report") or {}).get("checks", []) if not row["passed"]
    }
    if failures != set(recovery.get("archived_check_ids", [])) or not failures:
        raise ValueError("Archived failed-check IDs do not match the original validation report.")
    checks = {sample["id"]: sample for sample in original["samples"] if sample["role"] == "check"}
    if not failures.issubset(checks):
        raise ValueError("An archived failed check has no original observation.")
    for sample_id in sorted(failures):
        sample = checks[sample_id]
        if any(key not in sample for key in ("T_base_tool", "T_camera_board", "observation")):
            raise ValueError("An archived failed check is missing its measured geometry.")
        image = (parent / sample["image"]).resolve()
        if not image.is_relative_to(parent) or not image.is_file():
            raise ValueError("An archived failed-check image is missing or outside its run.")
        if hashlib.sha256(image.read_bytes()).hexdigest() != sample["image_sha256"]:
            raise ValueError("An archived failed-check image no longer matches its hash.")
        # Keep only immutable geometry/provenance, not large ROS feedback windows.
        held_out = {
            key: copy.deepcopy(sample[key])
            for key in (
                "id",
                "role",
                "T_base_tool",
                "T_camera_board",
                "observation",
                "image",
                "image_sha256",
            )
            if key in sample
        }
        held_out["source_run_id"] = parent.name
        samples.append(held_out)
    sources.append(
        {
            "source_run_id": parent.name,
            "source_run_path": str(parent),
            "source_run_sha256": digest,
            "failed_check_ids": sorted(failures),
        }
    )
    evidence = {
        "schema": 1,
        "mount_unchanged": True,
        "source_run_id": parent.name,
        "sources": sources,
        "samples": samples,
        "scope": "Prior failed validation views, held out permanently while mounting is unchanged.",
    }
    evidence["sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return evidence


def fork_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.parent != destination.parent or source == destination:
        raise ValueError("The new attempt must be a separate sibling run directory.")
    if destination.exists():
        raise ValueError("The destination already exists; no recordings were changed.")
    if (source / "archive.json").exists():
        raise ValueError("This run is already archived.")
    raw = (source / "run.json").read_bytes()
    original = json.loads(raw)
    accepted = [s for s in original["samples"] if s["role"] == "training" and not s.get("excluded")]
    if not accepted:
        raise ValueError("There is no accepted training checkpoint to resume.")
    failed = [c["id"] for c in (original.get("report") or {}).get("checks", []) if not c["passed"]]
    if not failed:
        raise ValueError("This recovery operation requires a run with failed checks.")
    files = ["arm_model.urdf", "stereo_params.npz"] + [s["image"] for s in accepted]
    for relative in files:
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("A checkpoint file is missing or outside the source run.")
    for sample in accepted:
        if (
            hashlib.sha256((source / sample["image"]).read_bytes()).hexdigest()
            != sample["image_sha256"]
        ):
            raise ValueError("A checkpoint image no longer matches its recorded hash.")

    created = datetime.now(UTC).isoformat()
    data = copy.deepcopy(original)
    data.update(
        created_at=created,
        samples=copy.deepcopy(accepted),
        stage="training",
        solution=None,
        report=None,
        policy=dict(POLICY),
    )
    data.pop("readiness", None)
    data["recovery"] = {
        "parent_run_id": source.name,
        "parent_run_path": str(source),
        "parent_run_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sample_id": accepted[-1]["id"],
        "retained_capture_ids": [s["id"] for s in accepted],
        "archived_check_ids": failed,
        "archived_check_count": len(failed),
        "mount_unchanged": True,
        "reason": "Operator requested restart from latest accepted capture; board and mount confirmed unchanged.",
        "validation_status": "Previous candidate failed. A new fit must satisfy its archived failed views and fresh checks; prior views never enter the fit.",
    }
    # Original capture provenance stays attached; code below will re-assess it.
    data["capture_source_sha256"] = original.get("source_sha256", {})
    data["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in original.get("source_sha256", {})
    }
    for sample in data["samples"]:
        sample["source_run_id"] = source.name
    data["prior_validation_evidence"] = load_prior_validation(data, destination)
    destination.mkdir()
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    (destination / "run.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (destination / "events.jsonl").write_text(
        json.dumps({"utc": created, "event": "checkpoint_forked", **data["recovery"]}) + "\n"
    )
    if (source / "run.json").read_bytes() != raw:
        raise RuntimeError(
            "Source changed during copying. Leave the new attempt paused and inspect both runs."
        )
    # A separate marker hides the failed attempt in normal resume choices while
    # retaining its original images, report, transform and run.json byte-for-byte.
    marker = {
        "utc": created,
        "reason": "failed_checks_archived_by_operator_request",
        "replacement_run_id": destination.name,
        "failed_check_ids": failed,
        "original_run_sha256": hashlib.sha256(raw).hexdigest(),
    }
    with (source / "archive.json").open("x") as stream:
        json.dump(marker, stream, indent=2)
        stream.write("\n")
    return data["recovery"]
