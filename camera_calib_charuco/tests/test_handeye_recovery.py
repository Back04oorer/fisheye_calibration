import hashlib
import json

import numpy as np
import pytest

from stereo_calibration.handeye_recovery import fork_checkpoint, load_prior_validation


def source_run(tmp_path):
    source = tmp_path / "source"
    (source / "frames").mkdir(parents=True)
    samples = []
    for i, role in enumerate(["training", "training", "check"]):
        image = f"frames/{i}.png"
        pixels = f"frame {i}".encode()
        (source / image).write_bytes(pixels)
        samples.append(
            {
                "id": str(i),
                "role": role,
                "image": image,
                "image_sha256": hashlib.sha256(pixels).hexdigest(),
                "T_base_tool": np.eye(4).tolist(),
                "T_camera_board": np.eye(4).tolist(),
                "observation": {},
            }
        )
    data = {
        "samples": samples,
        "solution": {"id": "bad-fit"},
        "report": {"passed": False, "checks": [{"id": "2", "passed": False}]},
        "stage": "checking",
    }
    (source / "run.json").write_text(json.dumps(data))
    (source / "arm_model.urdf").write_text("robot model")
    (source / "stereo_params.npz").write_text("camera model")
    (source / "result.json").write_text("failed report")
    return source


def test_fork_preserves_failed_evidence_and_requires_new_validation(tmp_path):
    source = source_run(tmp_path)
    originals = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "new"
    recovery = fork_checkpoint(source, destination)
    for relative, content in originals.items():
        assert (source / relative).read_bytes() == content
    data = json.loads((destination / "run.json").read_text())
    assert [s["id"] for s in data["samples"]] == ["0", "1"]
    assert data["solution"] is None and data["report"] is None and data["stage"] == "training"
    assert not (destination / "frames/2.png").exists()
    assert recovery["checkpoint_sample_id"] == "1"
    assert recovery["archived_check_ids"] == ["2"]
    assert data["prior_validation_evidence"]["samples"][0]["id"] == "2"
    assert data["prior_validation_evidence"]["samples"][0]["role"] == "check"
    assert load_prior_validation(data, destination) == data["prior_validation_evidence"]
    assert json.loads((source / "archive.json").read_text())["replacement_run_id"] == "new"
    with pytest.raises(ValueError, match="already archived"):
        fork_checkpoint(source, tmp_path / "another")


def test_corrupted_checkpoint_is_rejected_before_archiving(tmp_path):
    source = source_run(tmp_path)
    (source / "frames/1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        fork_checkpoint(source, tmp_path / "new")
    assert not (source / "archive.json").exists()
    assert not (tmp_path / "new").exists()


def test_legacy_recovery_reloads_hash_verified_failed_evidence(tmp_path):
    source = source_run(tmp_path)
    destination = tmp_path / "new"
    fork_checkpoint(source, destination)
    data = json.loads((destination / "run.json").read_text())
    del data["prior_validation_evidence"]
    del data["recovery"]["mount_unchanged"]
    restored = load_prior_validation(data, destination)
    assert restored["source_run_id"] == "source"
    assert restored["samples"][0]["id"] == "2"
    assert restored["mount_unchanged"] is True


@pytest.mark.parametrize("changed", ["run", "image", "mount", "model"])
def test_prior_evidence_tampering_or_setup_change_fails_closed(tmp_path, changed):
    source = source_run(tmp_path)
    destination = tmp_path / "new"
    fork_checkpoint(source, destination)
    data = json.loads((destination / "run.json").read_text())
    if changed == "run":
        path = source / "run.json"
        path.write_text(path.read_text() + " ")
    elif changed == "image":
        (source / "frames/2.png").write_bytes(b"different failure image")
    elif changed == "mount":
        data["recovery"]["mount_unchanged"] = False
    else:
        data["calibration_sha256"] = "different camera calibration"
    with pytest.raises(ValueError):
        load_prior_validation(data, destination)


def test_nested_restart_keeps_every_prior_failed_view(tmp_path):
    source = source_run(tmp_path)
    destination = tmp_path / "new"
    fork_checkpoint(source, destination)
    data = json.loads((destination / "run.json").read_text())
    image = b"new failed check"
    (destination / "frames/3.png").write_bytes(image)
    data["samples"].append(
        {
            "id": "3",
            "role": "check",
            "image": "frames/3.png",
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "T_base_tool": np.eye(4).tolist(),
            "T_camera_board": np.eye(4).tolist(),
            "observation": {},
        }
    )
    data["report"] = {"checks": [{"id": "3", "passed": False}]}
    (destination / "run.json").write_text(json.dumps(data))
    last = tmp_path / "last"
    fork_checkpoint(destination, last)
    data = json.loads((last / "run.json").read_text())
    evidence = load_prior_validation(data, last)
    assert [(s["source_run_id"], s["id"]) for s in evidence["samples"]] == [
        ("source", "2"),
        ("new", "3"),
    ]
    assert len(evidence["sources"]) == 2
