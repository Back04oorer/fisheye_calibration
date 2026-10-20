"""Independent synthetic geometry verifies coverage and predictive accuracy gates."""

import copy
import json

import cv2
import numpy as np
import pytest
from test_handeye import fixture_samples

from stereo_calibration import handeye


def test_coverage_is_order_and_duplicate_invariant_and_reports_translation():
    samples, _model, *_ = fixture_samples()
    poses = [s["T_base_tool"] for s in samples[:5]]
    expected = handeye.diversity(poses)
    actual = handeye.diversity(poses[::-1] + [poses[0]] * 100)
    assert expected["sufficient"] and actual["sufficient"]
    np.testing.assert_allclose(
        expected["rotation_rms_singular_values_rad"],
        actual["rotation_rms_singular_values_rad"],
        atol=1e-10,
    )
    assert actual["unique_rotation_count"] == 5
    assert actual["rotation_singular_values_normalized"]
    assert actual["translation_span_m"] == pytest.approx(expected["translation_span_m"])
    assert expected["translation_span_m"] > 0.1


def test_noisy_repeat_density_cannot_dilute_existing_rotation_coverage():
    poses = [
        np.eye(4),
        handeye.transform(cv2.Rodrigues(np.array([0.5, 0.0, 0.0]))[0], [0, 0, 0]),
        handeye.transform(cv2.Rodrigues(np.array([0.0, 0.5, 0.0]))[0], [0, 0, 0]),
    ]
    expected = handeye.diversity(poses)
    # Dense, translation-novel views with <0.12 degrees orientation jitter used
    # to dilute the second-axis score from 16.54 to 8.24 degrees and block it.
    repeated = poses + [
        handeye.transform(
            cv2.Rodrigues(np.array([1e-4 * (i + 1), 0.0, 0.0]))[0],
            [0.02 * (i % 5), 0.02 * (i // 5), 0],
        )
        for i in range(20)
    ]
    actual = handeye.diversity(repeated)
    assert actual["sufficient"] and actual["rotation_support_count"] == 3
    assert actual["second_axis_rms_deg"] == pytest.approx(expected["second_axis_rms_deg"], abs=0.12)
    shuffled = np.random.default_rng(37).permutation(len(repeated))
    reordered = handeye.diversity([repeated[i] for i in shuffled])
    assert actual == reordered
    # Duplicating all the noisy views changes no representative or score.
    dense = handeye.diversity(repeated + repeated[3:] * 10)
    assert dense["rotation_support_count"] == 3
    assert dense["rotation_rms_singular_values_rad"] == actual["rotation_rms_singular_values_rad"]


def test_complete_link_does_not_chain_adjacent_rotations_into_one_support():
    poses = [
        handeye.transform(cv2.Rodrigues(np.array([np.radians(angle), 0, 0]))[0], [0, 0, 0])
        for angle in [0, 6, 12]
    ]
    _poses, groups, _representatives, _unique = handeye._rotation_support(poses)
    assert len(groups) == 2
    for group in groups:
        for i in group:
            for j in group:
                assert (
                    handeye.difference(poses[i], poses[j])[1]
                    <= handeye.POLICY["pose_novelty_rotation_deg"]
                )


def test_a_hundred_single_axis_views_cannot_create_two_axis_coverage():
    poses = [
        handeye.transform(cv2.Rodrigues(np.array([i / 100, 0.0, 0.0]))[0], [i / 100, 0, 0])
        for i in range(100)
    ]
    coverage = handeye.diversity(poses)
    assert coverage["rotation_span_deg"] > 25
    assert coverage["translation_span_m"] > 0.9
    assert not coverage["sufficient"]


def test_mathematical_floor_is_not_a_completion_quota():
    samples, model, X, _Z = fixture_samples()
    three = handeye.training_assessment(samples[:3], model)
    assert three["solution"] is not None
    np.testing.assert_allclose(three["solution"]["T_base_camera_left_raw"], X, atol=1e-6)
    assert not three["ready"] and three["status"] == "stability"
    four = handeye.training_assessment(samples[:4], model)
    assert four["ready"]
    assert four["fit_max_rms_px"] < 0.001
    assert four["leave_one_out_max_rms_px"] < 0.001


def test_repeated_pose_does_not_supply_a_missing_held_out_orientation():
    samples, model, *_ = fixture_samples()
    repeated = copy.deepcopy(samples[:3])
    for index in range(20):
        duplicate = copy.deepcopy(samples[0])
        duplicate["id"] = f"duplicate-{index}"
        repeated.append(duplicate)
    result = handeye.training_assessment(repeated, model)
    assert not result["ready"] and result["status"] == "stability"


def test_micron_jitter_cannot_bypass_leaving_an_essential_orientation_out():
    samples, model, *_ = fixture_samples()
    repeated = copy.deepcopy(samples[:3])
    assert not handeye.training_assessment(repeated, model)["ready"]
    for sample in copy.deepcopy(repeated):
        sample["id"] += "-jitter"
        sample["T_base_tool"][0][3] += 1e-6
        repeated.append(sample)
    assessment = handeye.training_assessment(repeated, model)
    assert not assessment["ready"] and assessment["status"] == "stability"
    assert assessment["coverage"]["rotation_support_count"] == 3
    assert len(assessment["fit"]) == len(repeated)  # Every image is still tested.
    assert len(assessment["leave_one_out"]) == len(repeated)


def test_good_fit_can_fail_leave_one_out_prediction():
    samples, model, *_ = fixture_samples()
    # A 2 mm pairing error can be hidden by a full fit over four views; it must
    # not become a ready calibration when it fails to predict an omitted view.
    samples = copy.deepcopy(samples[:4])
    samples[0]["T_base_tool"][0][3] += 0.002
    result = handeye.training_assessment(samples, model)
    assert result["fit_max_rms_px"] < 3
    assert result["leave_one_out_max_rms_px"] > 3
    assert result["threshold_px"] == 3
    assert not result["ready"] and result["status"] == "stability"


def test_observable_fold_does_not_need_the_full_capture_envelope():
    from stereo_calibration.observations import Observation

    _samples, model, X, Z = fixture_samples()
    samples = []
    objects = handeye.L20_BOARD.create().getChessboardCorners().astype(float)
    for index, rotation in enumerate(
        [[-20, 0, 0], [-10, 0, 0], [10, 0, 0], [20, 0, 0], [0, -12, 0], [0, 12, 0]]
    ):
        C = handeye.transform(cv2.Rodrigues(np.radians(rotation))[0], [-0.04, -0.05, 0.5])
        right = handeye.transform(model.R, model.T / 1000) @ C
        points = [
            cv2.fisheye.projectPoints(
                objects.reshape(-1, 1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3] * 1000, K, D
            )[0].reshape(-1, 2)
            for T, K, D in [(C, model.K1, model.D1), (right, model.K2, model.D2)]
        ]
        observation = Observation(
            str(index), model.image_size, np.arange(12), points[0], np.arange(12), points[1]
        )
        samples.append(
            {
                "id": str(index),
                "role": "training",
                "T_camera_board": C.tolist(),
                "T_base_tool": (X @ C @ handeye.inverse(Z)).tolist(),
                "observation": observation.to_dict(),
            }
        )
    assert handeye.diversity([s["T_base_tool"] for s in samples])["sufficient"]
    reduced = handeye.diversity([s["T_base_tool"] for s in samples[:-1]])
    assert not reduced["sufficient"] and reduced["second_axis_rms_deg"] > 7
    assessment = handeye.training_assessment(samples, model)
    assert assessment["ready"]
    assert assessment["leave_one_out_max_rms_px"] < 0.001
    assert assessment["threshold_px"] == 3

    # Removing the sole remaining second-axis pose really is unobservable.
    insufficient = handeye.training_assessment(samples[:-1], model)
    assert not insufficient["ready"]


def test_more_samples_do_not_override_bad_native_pixel_accuracy():
    samples, model, *_ = fixture_samples()
    samples[0]["T_base_tool"][0][3] += 0.05
    result = handeye.training_assessment(samples, model)
    assert result["coverage"]["sufficient"]
    assert result["fit_max_rms_px"] > 3
    assert not result["ready"] and result["status"] == "accuracy"


def test_provisional_bad_fit_requests_redundant_tilts_without_relaxing_acceptance():
    from stereo_calibration.handeye_guidance import requires_review

    samples, model, *_ = fixture_samples()
    samples[0]["T_base_tool"][0][3] += 0.05
    early = copy.deepcopy(samples[:3])
    for sample in copy.deepcopy(early):
        sample["id"] += "-repeat"
        early.append(sample)
    before = copy.deepcopy(early)
    provisional = handeye.training_assessment(early, model)
    assert provisional["coverage"]["sufficient"]
    assert provisional["fit_max_rms_px"] > provisional["threshold_px"] == 3
    assert not provisional["cross_validation_coverage_sufficient"]
    assert not provisional["ready"] and provisional["status"] == "stability"
    assert not requires_review(provisional)
    assert "provisional fit" in provisional["reason"]
    assert "above the 3.0 px limit" in provisional["reason"]
    assert len(provisional["fit"]) == 6 and early == before

    # Adding actual orientation support enables a firm accuracy assessment;
    # it does not make the same inconsistent measurements acceptable.
    supported = handeye.training_assessment(samples, model)
    assert supported["cross_validation_coverage_sufficient"]
    assert not supported["ready"] and supported["status"] == "accuracy"
    assert supported["fit_max_rms_px"] > supported["threshold_px"] == 3
    assert requires_review(supported)


def test_independent_check_data_never_enters_training_assessment():
    samples, model, *_ = fixture_samples()
    baseline = handeye.training_assessment(samples, model)
    for sample in samples[12:]:
        sample["T_base_tool"] = "deliberately unreadable held-out pose"
        sample["observation"] = "deliberately unreadable held-out image"
    actual = handeye.training_assessment(samples, model)
    assert actual == baseline


def test_fresh_check_coverage_replaces_fixed_count_and_failure_cannot_be_hidden():
    samples, model, *_ = fixture_samples()
    solution = handeye.solve(samples)
    assert not handeye.validation_report(samples[:14], solution, model)["passed"]
    good = handeye.validation_report(samples[:15], solution, model)
    assert good["passed"] and len(good["checks"]) == 3
    assert not good["required_checks_is_observability_floor"]
    assert "no transform is estimated" in good["required_checks_basis"]
    # Exclusion is not a mechanism to pass the SAME frozen candidate.
    samples[12]["T_base_tool"][0][3] += 0.05
    samples[12]["excluded"] = True
    failed = handeye.validation_report(samples, solution, model)
    assert not failed["passed"]
    assert not next(row for row in failed["checks"] if row["id"] == "12")["passed"]


def test_a_count_of_good_checks_without_orientation_coverage_does_not_pass():
    samples, model, *_ = fixture_samples()
    solution = handeye.solve(samples)
    checks = [copy.deepcopy(samples[12]) for _ in range(50)]
    for i, check in enumerate(checks):
        check["id"] = f"same-check-{i}"
    report = handeye.validation_report(samples[:12] + checks, solution, model)
    assert all(row["passed"] for row in report["checks"])
    assert not report["passed"] and not report["coverage"]["sufficient"]


def test_isolated_opencv_batches_cross_validation_in_one_process(monkeypatch):
    samples, model, *_ = fixture_samples()
    calls = []
    original = handeye.subprocess.run

    def record(*args, **kwargs):
        calls.append(json.loads(kwargs["input"]))
        return original(*args, **kwargs)

    monkeypatch.delattr(handeye.cv2, "calibrateHandEye", raising=False)
    monkeypatch.setattr(handeye.subprocess, "run", record)
    assert handeye.training_assessment(samples[:4], model)["ready"]
    assert len(calls) == 1
    assert len(calls[0]["batches"]) == 5


def prior_evidence(sample):
    return {
        "mount_unchanged": True,
        "source_run_id": "archived",
        "samples": [copy.deepcopy(sample)],
    }


def test_identical_refit_cannot_pass_by_archiving_an_unresolved_failed_check():
    samples, model, *_ = fixture_samples()
    train = samples[:12]
    failed = copy.deepcopy(samples[12])
    failed["T_base_tool"][0][3] += 0.05
    source_solution = handeye.solve(train + [failed], model)
    # Fresh passing views alone would let exactly the same transform pass.
    fresh = train + samples[13:17]
    refit = handeye.solve(fresh, model)
    assert refit["T_base_camera_left_raw"] == source_solution["T_base_camera_left_raw"]
    assert handeye.validation_report(fresh, refit, model)["passed"]
    evidence = prior_evidence(failed)
    assessment = handeye.training_assessment(fresh, model, prior_validation=evidence)
    assert not assessment["ready"] and assessment["status"] == "prior_validation"
    assert assessment["prior_validation"]["max_rms_px"] > 3
    assert assessment["solution"] == refit  # Prior check did not enter the fit.
    report = handeye.validation_report(fresh, refit, model, prior_validation=evidence)
    assert all(check["passed"] for check in report["checks"])
    assert not report["passed"] and not report["prior_validation"]["passed"]
    assert len(report["checks"]) == 4  # Historical evidence is not active capture count.


def test_consistent_prior_evidence_still_requires_fresh_validation():
    samples, model, *_ = fixture_samples()
    evidence = prior_evidence(samples[12])
    assessment = handeye.training_assessment(samples[:12], model, prior_validation=evidence)
    assert assessment["ready"] and assessment["prior_validation"]["passed"]
    result = handeye.validation_report(
        samples[:12], assessment["solution"], model, prior_validation=evidence
    )
    assert not result["passed"] and result["prior_validation"]["passed"]
    assert handeye.validation_report(
        samples, assessment["solution"], model, prior_validation=evidence
    )["passed"]
