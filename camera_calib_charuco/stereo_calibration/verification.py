"""Re-evaluate saved check evidence and plan only the outstanding regions.

Capture provenance stays immutable. A region passes only under the current model,
and an old failed measurement cannot be replaced by a later favourable capture.
"""

import cv2

from .guidance import VALIDATION_GOALS, achieved, describe
from .observations import Observation
from .validation import evaluate


def evaluate_saved_checks(captures, model, board, candidate_id, required_fresh=()):
    observations = [Observation.from_dict(capture["observation"]) for capture in captures]
    report = evaluate(observations, model, board)
    passed, failed, fresh, reused = set(), set(), set(), set()
    keys = {goal.key for goal in VALIDATION_GOALS}
    for capture, observation, view in zip(captures, observations, report["validation_views"]):
        try:
            signature = describe(observation, board, model)
            goals = set(achieved(signature, None, validation=True))
        except (cv2.error, ValueError):
            signature, goals = None, set()
        if signature is None:
            view["passed"] = False
            view.setdefault("error", "Board position could not be estimated for coverage.")
        if not view["passed"]:
            # Retain the recorded region as a retry hint if pose estimation failed.
            goals.update(keys.intersection(capture["goals"]))
            failed.update(goals)
        else:
            passed.update(goals)
            (fresh if capture["candidate"] == candidate_id else reused).update(goals)
        view.update(
            goals=sorted(goals),
            capture_candidate=capture["candidate"],
            reused=capture["candidate"] != candidate_id,
        )
    pending_confirmation = keys.intersection(required_fresh) - fresh
    report["regions"] = [
        {
            "key": goal.key,
            "title": goal.title.removeprefix("Check "),
            "status": "failed"
            if goal.key in failed
            else "confirm"
            if goal.key in pending_confirmation
            else "passed"
            if goal.key in passed
            else "missing",
        }
        for goal in VALIDATION_GOALS
    ]
    report["goals_complete"] = all(region["status"] == "passed" for region in report["regions"])
    report["passed"] = bool(
        report["fit_passed"]
        and report["goals_complete"]
        and report["validation_views"]
        and all(view["passed"] for view in report["validation_views"])
    )
    report["reused_passed_regions"] = [
        region["key"]
        for region in report["regions"]
        if region["status"] == "passed" and region["key"] in reused
    ]
    report["required_fresh_goals"] = sorted(keys.intersection(required_fresh))
    report["capture_ids"] = [capture["id"] for capture in captures]
    report["verification_policy"] = "all_saved_checks_v1"
    report["scope"] = (
        "Geometric consistency on saved check images, kept separate from fitting. "
        "Checks reused during refinement are not an untouched statistical test set. "
        "Physical depth accuracy is not measured."
    )
    return report
