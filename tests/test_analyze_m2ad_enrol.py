"""Stage 7 analysis on hand-made arrays (no torch, no data)."""

import json

import numpy as np
import pytest

from defect_inspect import analyze_m2ad_enrol as an
from defect_inspect.m2ad import ILLUMINATIONS
from defect_inspect.run_m2ad_enrol import LOOP_ARMS, enrolled_matrix, loop_keys

N_SPEC = 10


@pytest.fixture(autouse=True)
def no_real_outputs(tmp_path, monkeypatch):
    """The stage 3-B comparison must not find the real runs of this machine."""
    monkeypatch.setattr(an.paths, "OUTPUTS", tmp_path / "outputs")


INSPECTORS = ["Motor_000", "Motor_120", "Bird_000"]


def _val_arrays(candidate: str, flag_unseen: float, seed: int = 0) -> dict:
    """Validation arrays where a share `flag_unseen` of unseen images (and none of the seen) exceed."""
    rng = np.random.default_rng(seed)
    enrolled = enrolled_matrix(candidate)
    n_arms = len(enrolled)
    folds = np.arange(N_SPEC) % 5
    thresholds = np.ones((n_arms, 5))
    scores = np.full((n_arms, 10, N_SPEC), 0.5, dtype=np.float32)
    for a in range(n_arms):
        for li in range(10):
            if not enrolled[a, li]:
                scores[a, li] = np.where(rng.random(N_SPEC) < flag_unseen, 2.0, 0.5)
    return {
        "specimens": np.array([f"s{i:02d}" for i in range(N_SPEC)]),
        "folds": folds,
        "lights": np.array(ILLUMINATIONS),
        "arms": np.array(["ref"] if n_arms == 1 else ["A", "B"]),
        "enrolled": enrolled,
        "scores": scores,
        "thresholds": thresholds,
        "cal_n": np.full((n_arms, 5), 8),
    }


def _val_run(shares: dict[str, float]) -> dict:
    return {
        "meta": {"commit": "abc", "device": "cpu", "groups": None, "inspectors": INSPECTORS},
        "cands": {
            c: {n: _val_arrays(c, share, seed=i) for i, n in enumerate(INSPECTORS)}
            for c, share in shares.items()
        },
    }


def test_validation_counts_unseen_images_of_every_arm():
    run = _val_run({"e0": 1.0, "e2": 0.0})
    report = an.build_val_report({"p0": run}, n_boot=50)
    rows = report["methods"]["p0"]["candidates"]
    # Every candidate has the same unseen images: 9 lights x specimens x inspectors.
    assert rows["e0"]["unseen"]["n_normal"] == rows["e2"]["unseen"]["n_normal"] == 9 * N_SPEC * 3
    assert rows["e0"]["unseen"]["fpr"] == 1.0 and rows["e2"]["unseen"]["fpr"] == 0.0
    assert rows["e0"]["seen"]["n_normal"] == N_SPEC * 3
    assert rows["e2"]["seen"]["n_normal"] == (5 + 6) * N_SPEC * 3  # I01 counts once in each arm
    assert report["methods"]["p0"]["pick"] == "e2"
    assert set(rows["e2"]["unseen_by_light"]) == set(ILLUMINATIONS[1:])


def test_pick_ties_go_to_the_simpler_candidate():
    rows = {
        c: {"unseen": {"false_positives": fp, "n_normal": 100}} for c, fp in (("e3", 5), ("e1", 5), ("e2", 6))
    }
    assert an.choose(rows) == "e1"
    rows["e0"] = {"unseen": {"false_positives": 5, "n_normal": 100}}
    assert an.choose(rows) == "e0"
    rows["e3"]["unseen"]["false_positives"] = 4
    assert an.choose(rows) == "e3"


def test_validation_interval_resamples_specimens_per_category():
    run = _val_run({"e0": 0.5})
    report = an.build_val_report({"p0": run}, n_boot=200)
    lo, hi = report["methods"]["p0"]["candidates"]["e0"]["unseen"]["fpr_ci"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    bad = _val_run({"e0": 0.5})
    bad["cands"]["e0"]["Motor_120"]["specimens"] = bad["cands"]["e0"]["Motor_120"]["specimens"][::-1].copy()
    with pytest.raises(ValueError):
        an.build_val_report({"p0": bad}, n_boot=10)


# ---------------------------------------------------------------- test


N_NORMAL, N_DEFECT = 4, 3


def _test_arrays(candidate: str, unseen_normal_score: float, ref_defect_score: float = 3.0) -> dict:
    enrolled = enrolled_matrix(candidate)
    n_arms, n = len(enrolled), N_NORMAL + N_DEFECT
    labels = np.array([[0] * N_NORMAL + [1] * N_DEFECT] * 10, dtype=np.int8)
    labels[:, -1] = -1  # one defect never visible
    scores = np.zeros((n_arms, 10, n), dtype=np.float32)
    for a in range(n_arms):
        for li in range(10):
            normal = unseen_normal_score if not enrolled[a, li] else 0.5
            scores[a, li, :N_NORMAL] = normal
            scores[a, li, N_NORMAL:] = ref_defect_score if li == 0 else 3.0
    return {
        "specimens": np.array([f"n{i}" for i in range(N_NORMAL)] + [f"d{i}" for i in range(N_DEFECT)]),
        "object_anomaly": np.array([0] * N_NORMAL + [1] * N_DEFECT, dtype=np.int8),
        "lights": np.array(ILLUMINATIONS),
        "arms": np.array(["ref"] if n_arms == 1 else ["A", "B"]),
        "enrolled": enrolled,
        "scores": scores,
        "labels": labels,
        "thresholds": np.ones(n_arms),
    }


def _test_run(pick: str, pick_unseen: float, pick_ref_defect: float = 3.0, loop: bool = False) -> dict:
    cands = {"e0": {n: _test_arrays("e0", 2.0) for n in INSPECTORS}}
    if pick != "e0":
        cands[pick] = {n: _test_arrays(pick, pick_unseen, pick_ref_defect) for n in INSPECTORS}
    meta = {"commit": "abc", "device": "cpu", "candidates": list(cands), "inspectors": INSPECTORS}
    run = {"meta": meta, "cands": cands, "loop": None}
    if loop:
        keys = loop_keys()
        labels = np.array([[0] * N_NORMAL + [1] * N_DEFECT] * len(keys), dtype=np.int8)
        scores = np.tile(np.array([0.5] * N_NORMAL + [3.0] * N_DEFECT, dtype=np.float32), (len(keys), 1))
        run["loop"] = {
            n: {
                "loop_keys": np.array(keys),
                "scores": scores,
                "labels": labels,
                "thresholds": np.ones(len(keys)),
            }
            for n in INSPECTORS
        }
        info = {
            key: {
                "defects_in_batch": int(key.split(":")[2]),
                "dropped_defects": int(key.split(":")[2]),
                "dropped_normals": 4 - int(key.split(":")[2]) if "trim" in key else 0,
            }
            for key in keys
        }
        meta["loop_runs"] = [
            {"category": n.split("_")[0], "view": n.split("_")[1], "loop": info} for n in INSPECTORS
        ]
    return run


def test_h18_supported_when_the_pick_removes_the_unseen_false_alarms():
    report = an.build_test_report({"p0": _test_run("e2", 0.5), "d-s": _test_run("e2", 0.5)}, n_boot=100)
    v = report["methods"]["p0"]["verdicts"]
    assert v["pick"] == "e2" and v["d_unseen_fpr"] == -1.0 and v["H18"] == "지지"
    assert v["guard_ok"] and v["improved"] and v["H19"] == "지지" and v["seen_fpr"] == 0.0
    assert report["hypotheses"] == {"H18": "지지", "H19": "지지"}
    # The second method is reported without verdicts.
    assert (
        "H18" not in report["methods"]["d-s"]["verdicts"]
        and "improved" not in report["methods"]["d-s"]["verdicts"]
    )
    cand = report["methods"]["p0"]["candidates"]["e2"]
    assert cand["unseen"]["n_normal"] == 9 * N_NORMAL * 3 and cand["unseen"]["n_defect"] == 9 * 2 * 3
    assert cand["seen"]["n_normal"] == 11 * N_NORMAL * 3
    assert cand["reference"]["n_defect"] == 2 * 2 * 3  # I01 in both arms
    assert report["methods"]["p0"]["stage3b_match"] is None
    json.dumps(an.strict_json(report))


def test_the_detection_guard_and_the_other_verdicts():
    # The pick misses every defect under I01: the false alarms go, but it is no improvement.
    report = an.build_test_report({"p0": _test_run("e3", 0.5, pick_ref_defect=0.5)}, n_boot=50)
    v = report["methods"]["p0"]["verdicts"]
    assert v["H18"] == "지지" and v["d_reference_tpr"] == -1.0 and not v["guard_ok"] and not v["improved"]
    # Half the unseen normals still flagged: the drop is 50 points at most, so H18 is rejected.
    report = an.build_test_report({"p0": _test_run("e1", 2.0)}, n_boot=50)
    assert report["methods"]["p0"]["verdicts"]["H18"] == "기각"
    # The pick was e0 itself: nothing to compare, rejected.
    report = an.build_test_report({"p0": _test_run("e0", 0.0)}, n_boot=50)
    v = report["methods"]["p0"]["verdicts"]
    assert v["pick"] == "e0" and v["H18"] == "기각" and v["H19"] == "지지"


def test_loop_table_pools_the_nine_illuminations():
    report = an.build_test_report({"p0": _test_run("e2", 0.5, loop=True)}, n_boot=20)
    rows = report["methods"]["p0"]["loop"]
    assert [(r["filter"], r["k"]) for r in rows] == [tuple(arm) for arm in LOOP_ARMS]
    for row in rows:
        assert row["n_normal"] == 9 * N_NORMAL * 3 and row["fpr"] == 0.0 and row["tpr"] == 1.0
        assert row["defects_in_batches"] == 9 * 3 * row["k"]


def test_stage3b_match(tmp_path):
    run = _test_run("e0", 0.0)
    ref = tmp_path / "m2ad-p0"
    ref.mkdir()
    (ref / "run.json").write_text("{}", encoding="utf-8")
    conditions = ["S", *(f"R:{light}" for light in ILLUMINATIONS[1:])]
    for name, arr in run["cands"]["e0"].items():
        np.savez(
            ref / f"{name}.npz",
            conditions=np.array(conditions),
            scores=arr["scores"][0],
            threshold=np.float64(1.0),
        )
    assert an.stage3b_match(run, ref) == {"max_abs_score_diff": 0.0, "thresholds_equal": True}
    assert an.stage3b_match(run, tmp_path / "missing") is None
    np.savez(ref / "Bird_000.npz", specimens=np.array(["x"]), conditions=np.array(conditions))
    with pytest.raises(ValueError, match="other test specimens"):
        an.stage3b_match(run, ref)


# ---------------------------------------------------------------- VisA dev


def _perturb_run(shift: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cats = {}
    for c in ("candle", "pcb1"):
        labels = np.array([0] * 30 + [1] * 10, dtype=np.int8)
        clean = np.concatenate([rng.normal(0, 1, 30), rng.normal(2, 1, 10)])
        scores = np.stack([clean, clean + shift, clean + 0.5]).astype(np.float32)
        cats[c] = {
            "eval_images": np.array([f"{c}/{i}.JPG" for i in range(40)]),
            "eval_labels": labels,
            "conditions": np.array(an.VISA_CONDITIONS),
            "scores": scores,
            "cal_score": rng.normal(0, 1, 60).astype(np.float32),
        }
    return {"meta": {"protocol": "dev", "commit": "x"}, "cats": cats}


def test_visa_dev_table_pairs_the_two_runs():
    e0, e1 = _perturb_run(shift=1.0), _perturb_run(shift=0.0)
    table = an.visa_dev_table(e0, e1, n_boot=100)
    assert [row["condition"] for row in table] == list(an.VISA_CONDITIONS)
    clean, bright = table[0], table[1]
    # Same clean scores: no difference at all, and a uniform shift leaves the AUROC alone.
    assert clean["d_auroc"] == 0.0 and clean["d_fpr"] == 0.0 and clean["d_auroc_ci"] == [0.0, 0.0]
    assert bright["e0_auroc"] == bright["e1_auroc"] and bright["e0_d_auroc_vs_clean"] == 0.0
    assert bright["e0_fpr"] > bright["e1_fpr"] and bright["d_fpr"] < 0 and bright["d_fpr_ci"][1] < 0
    other = _perturb_run(shift=0.0)
    other["cats"]["pcb1"]["eval_images"] = other["cats"]["pcb1"]["eval_images"][::-1].copy()
    with pytest.raises(ValueError):
        an.visa_dev_table(e0, other, n_boot=10)
