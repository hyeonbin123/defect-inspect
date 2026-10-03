import json
import math

import numpy as np
import pytest

from defect_inspect import drift


def test_inductive_pvalues_count_larger_and_tied_calibration_scores():
    cal = np.array([1.0, 2.0, 2.0, 3.0])
    scores = np.array([[0.5, 2.0, 3.5]])
    theta = np.array([[1.0, 0.5, 0.25]])
    p = drift.inductive_pvalues(scores, cal, theta)
    # 0.5: all four are larger, (4 + 1 * 1) / 5; 2.0: one larger, two tied, (1 + 0.5 * 3) / 5; 3.5: none.
    np.testing.assert_allclose(p, [[1.0, 0.5, 0.05]])
    with pytest.raises(ValueError):
        drift.inductive_pvalues(scores, np.array([]), theta)


def test_transductive_pvalues_rank_each_score_among_those_seen_so_far():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=(7, 30)).astype(float)  # many ties
    theta = drift._theta(rng, scores.shape)
    p = drift.transductive_pvalues(scores, theta)
    for r in range(scores.shape[0]):
        for t in range(scores.shape[1]):
            past = scores[r, : t + 1]
            expected = ((past > scores[r, t]).sum() + theta[r, t] * (past == scores[r, t]).sum()) / (t + 1)
            assert p[r, t] == pytest.approx(expected)
    # The first observation is compared with itself only.
    np.testing.assert_allclose(p[:, 0], theta[:, 0])
    assert ((p > 0) & (p <= 1)).all()


def test_transductive_pvalues_are_uniform_for_exchangeable_scores():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(400, 50))
    p = drift.transductive_pvalues(scores, drift._theta(rng, scores.shape))
    assert abs(p.mean() - 0.5) < 0.01
    assert abs((p < 0.1).mean() - 0.1) < 0.01


def test_the_martingale_rarely_alarms_on_uniform_pvalues():
    rng = np.random.default_rng(2)
    p = drift._theta(rng, (3000, 400))
    alarm = drift.first_alarm(p)
    # Ville: at most DELTA over an infinite stream.
    assert (alarm >= 0).mean() <= drift.DELTA


def test_the_martingale_alarms_soon_after_the_pvalues_turn_small():
    p = np.full((2, 60), 0.5)
    p[0, 40:] = 1e-4
    p[1, 40:] = 0.03
    alarm = drift.first_alarm(p)
    assert 40 <= alarm[0] <= 43 and 40 <= alarm[1] <= 50
    # Without the jump, the best function would not have lost so much during the quiet stretch.
    assert drift.first_alarm(np.full((1, 3), 1e-12))[0] in (0, 1, 2)
    assert drift.first_alarm(np.full((1, 100), 0.5))[0] == -1
    with pytest.raises(ValueError):
        drift.first_alarm(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        drift.first_alarm(np.full((1, 3), 0.5), epsilons=(1.0, 0.0))


def test_the_alarm_follows_the_capital_by_hand():
    p = np.array([[0.2, 0.001, 0.001, 0.001]])
    eps = np.array(drift.EPSILONS)
    capital = np.full(eps.size, 1 / eps.size)
    reached = -1
    for t, value in enumerate(p[0]):
        capital = (1 - drift.JUMP) * capital + drift.JUMP * capital.sum() / eps.size
        capital = capital * eps * value ** (eps - 1)
        if capital.sum() >= 1 / drift.DELTA and reached < 0:
            reached = t
    assert reached >= 0 and drift.first_alarm(p)[0] == reached
    assert drift.first_alarm(p, delta=1e-9)[0] == -1  # a much higher bar is not reached in four steps


def test_n_mixed():
    assert drift.n_mixed(200, 0.0, 40) == 0
    assert drift.n_mixed(200, 0.05, 40) == 11  # 10.53 rounds to 11
    assert drift.n_mixed(200, 0.10, 40) == 22
    assert drift.n_mixed(402, 0.10, 40) == 40  # capped by the defects there are
    assert drift.n_mixed(20, 0.05, 50) == 1 and drift.n_mixed(20, 0.10, 50) == 2
    with pytest.raises(ValueError):
        drift.n_mixed(10, 1.0, 5)


def test_false_alarm_streams_hold_every_normal_once():
    rng = np.random.default_rng(3)
    normal, defect = np.arange(20.0), np.arange(100.0, 110.0)
    streams, k = drift.false_alarm_streams(normal, defect, 0.10, 5, rng)
    assert k == 2 and streams.shape == (5, 22)
    for row in streams:
        assert sorted(row[row < 100]) == list(normal) and (row >= 100).sum() == 2
    assert len({tuple(row) for row in streams}) == 5  # each stream has its own order


def test_split_change_streams_keep_images_apart():
    rng = np.random.default_rng(4)
    labels = np.array([0] * 10 + [1] * 4)
    clean = np.arange(14.0)
    changed = clean + 1000.0
    streams, change = drift.split_change_streams(clean, changed, labels, 0.10, 6, rng)
    assert change == 5 + 1 and streams.shape == (6, 6 + 5 + 1)
    for row in streams:
        pre, post = row[:change], row[change:]
        assert (pre < 1000).all() and (post >= 1000).all()
        assert not set(pre.astype(int)) & set((post - 1000).astype(int))  # no image on both sides


def test_change_streams_and_delays():
    rng = np.random.default_rng(5)
    pre = (np.zeros(20), np.full(5, 3.0))
    post = (np.full(20, 9.0), np.full(5, 12.0))
    streams, change, mixed = drift.change_streams(pre, post, 0.05, 4, rng)
    assert mixed == (1, 1) and change == 21 and streams.shape == (4, 42)
    early, late = drift.delays(np.array([3, 21, 25, -1]), change)
    assert early == 1 and late.tolist() == [1.0, 5.0, math.inf]
    summary = drift.pool_delays([(np.array([3, 21, 25, -1]), change), (np.array([30, 31]), 30)])
    assert summary["streams"] == 6 and summary["early_share"] == pytest.approx(1 / 6)
    assert summary["alarmed_share"] == pytest.approx(4 / 5) and summary["median_delay"] == 2.0
    assert drift.pool_delays([(np.array([-1, -1, 5]), 2)])["median_delay"] is None


def test_detect_runs_both_detectors_on_the_same_streams():
    rng = drift.rng_for("x")
    assert drift.rng_for("x").random() == rng.random()  # fixed seed per key
    scores = np.concatenate([np.zeros((50, 100)), np.full((50, 30), 5.0)], axis=1)
    scores += np.random.default_rng(6).normal(0, 0.1, scores.shape)
    found = drift.detect(scores, np.random.default_rng(7).normal(0, 0.1, 300), drift.rng_for("y"))
    assert set(found) == {"inductive", "transductive"}
    for alarm in found.values():
        assert ((alarm >= 100) & (alarm < 130)).mean() > 0.9
    assert set(drift.detect(scores, None, drift.rng_for("y"))) == {"transductive"}


def _visa_dir(root, run, score_key, cal_key):
    d = root / run
    d.mkdir(parents=True)
    rng = np.random.default_rng(8)
    for cat in ("candle", "pcb1"):
        labels = np.array([0] * 40 + [1] * 6, dtype=np.int8)
        np.savez(
            d / f"{cat}.npz",
            eval_labels=labels,
            **{score_key: rng.normal(size=46) + 3 * labels, cal_key: rng.normal(size=80)},
        )


def test_tables_from_saved_scores(tmp_path):
    for run, score_key, cal_key in drift.VISA_SOURCES.values():
        _visa_dir(tmp_path, run, score_key, cal_key)
    table = drift.fa_table("p0", tmp_path, n_streams=20)
    assert set(table) == {"0.00", "0.05", "0.10"}
    assert table["0.00"]["inductive"]["streams"] == 40 and table["0.10"]["mixed_defects"]["candle"] == 4
    assert set(table["0.05"]["per_category"]) == {"candle", "pcb1"}

    perturb = tmp_path / "perturb-p0-test"
    perturb.mkdir()
    rng = np.random.default_rng(9)
    labels = np.array([0] * 40 + [1] * 6, dtype=np.int8)
    clean = rng.normal(size=46)
    np.savez(
        perturb / "candle.npz",
        conditions=np.array(["clean", "brightness-3"]),
        scores=np.stack([clean, clean + 10]),
        eval_labels=labels,
        cal_score=rng.normal(size=80),
    )
    delay = drift.perturb_delay_table("p0", tmp_path, n_streams=10)
    row = delay["brightness-3"]["0.00"]
    assert row["inductive"]["alarmed_share"] == 1.0 and row["inductive"]["median_delay"] <= 5

    m2ad = tmp_path / "m2ad-p0"
    m2ad.mkdir()
    (m2ad / "run.json").write_text(
        json.dumps({"inspectors": [{"category": "Motor", "view": "000"}]}), "utf-8"
    )
    labels = np.array([[0] * 20 + [1] * 5 + [-1]] * 2, dtype=np.int8)
    np.savez(
        m2ad / "Motor_000.npz",
        conditions=np.array(["S", "R:02"]),
        scores=np.stack([rng.normal(size=26), rng.normal(size=26) + 10]),
        labels=labels,
        cal_score=rng.normal(size=30),
    )
    table = drift.m2ad_delay_table("p0", tmp_path, n_streams=10)
    assert set(table) == {"02"} and table["02"]["0.10"]["inductive"]["streams"] == 10
