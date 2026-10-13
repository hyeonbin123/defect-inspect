# [파일 역할] 7단계 라벨 없는 변화 감지 모듈(`defect_inspect.drift`)을 검증하는 테스트 묶음이다. 이미지 점수
# 스트림을 conformal p-value(귀납형: 보정 점수와 비교, 전도형: 스트림 자체의 순위)로 바꾸고, 거듭제곱 베팅
# 함수의 Simple Jumper 마팅게일 자본이 1/δ(=100)에 닿으면 경보하는 코드다.
# [왜 필요한가] p-value 공식, 동점 처리(θ), 베팅·재분배(J = 0.01) 순서, 결함 섞는 장수(round(r / (1 − r) ×
# 정상 수), 결함 수가 상한)는 모두 docs/experiments.md 7단계 '라벨 없는 변화 감지'에 측정 전에 등록됐다.
# 공식이 조금만 달라도 허위 경보율과 경보 지연 수치가 바뀐다. 또 Ville 부등식이 말하는 '교환 가능한 점수면
# 허위 경보 확률 δ 이하'가 구현에서도 성립하는지 모의로 본다.
# [흐름] p-value 두 가지(공식·균등성) → 마팅게일(허위 경보, 변화 뒤 경보, 손 계산과 대조) → 스트림 만들기(결함
# 섞기, 한 이미지가 변화 앞뒤에 같이 나오지 않기) → 지연 요약 → 저장된 점수 파일로 표 만들기.

import json
import math

import numpy as np
import pytest

from defect_inspect import drift


# 확인: 귀납형 p = (#{보정 > s} + θ(#{보정 = s} + 1)) / (n + 1)를 손으로 계산한 값과 맞춘다. 보정 점수보다
# 아주 작은 점수는 1, 동점이 있으면 θ로 나누어 갖고, 모두보다 큰 점수는 θ/(n+1)까지 작아진다. 보정 점수가 비면
# ValueError다.
# 왜: 점수가 클수록 이상이라 새 점수가 클수록 p가 작아야 한다(7단계 규칙). 동점을 θ로 무작위로 나누지 않으면
# 점수에 동점이 많은 방법에서 p-value가 균등하지 않게 된다.
def test_inductive_pvalues_count_larger_and_tied_calibration_scores():
    cal = np.array([1.0, 2.0, 2.0, 3.0])
    scores = np.array([[0.5, 2.0, 3.5]])
    theta = np.array([[1.0, 0.5, 0.25]])
    p = drift.inductive_pvalues(scores, cal, theta)
    # 0.5: all four are larger, (4 + 1 * 1) / 5; 2.0: one larger, two tied, (1 + 0.5 * 3) / 5; 3.5: none.
    np.testing.assert_allclose(p, [[1.0, 0.5, 0.05]])
    with pytest.raises(ValueError):
        drift.inductive_pvalues(scores, np.array([]), theta)


# 확인: 벡터화한 전도형 p-value가 정의(지금까지 본 점수 가운데 더 큰 수 + θ × 같은 수, 자기 자신 포함, t로
# 나눔)를 반복문으로 직접 계산한 값과 같다. 0~4 정수 점수로 동점을 많이 만들고, 첫 관측은 자기 자신과만
# 비교하므로 p = θ이며, 모든 값이 (0, 1]에 든다.
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


# 확인: 교환 가능한(무작위 순서의 정규분포) 점수에서 전도형 p-value의 평균이 0.5, 0.1보다 작은 비율이 10%로
# 균등분포처럼 나온다. 마팅게일의 허위 경보 보장이 이 균등성에 기대고 있다.
def test_transductive_pvalues_are_uniform_for_exchangeable_scores():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(400, 50))
    p = drift.transductive_pvalues(scores, drift._theta(rng, scores.shape))
    assert abs(p.mean() - 0.5) < 0.01
    assert abs((p < 0.1).mean() - 0.1) < 0.01


# 확인: 균등한 p-value 400개짜리 스트림 3,000개에서 한 번이라도 경보가 난 비율이 δ(1%) 이하다. Ville 부등식은
# 무한한 스트림 전체에 대해 경보 확률을 δ로 묶는다(설계 목표: 결함 섞임 0%에서 허위 경보 1% 이하).
def test_the_martingale_rarely_alarms_on_uniform_pvalues():
    rng = np.random.default_rng(2)
    p = drift._theta(rng, (3000, 400))
    alarm = drift.first_alarm(p)
    # Ville: at most DELTA over an infinite stream.
    assert (alarm >= 0).mean() <= drift.DELTA


# 확인: p가 0.5로 조용하다가 40번째부터 작아지면 곧 경보한다(1e-4면 4장 안, 0.03이면 11장 안). 처음부터 p가
# 아주 작으면 세 장 안에 경보하고, 0.5만 100장이면 경보하지 않는다. p가 0이거나 ε가 0이면 ValueError다.
# 왜: 조용한 구간에서 크게 건 함수는 자본을 잃는데, 베팅 전에 전체 자본의 J를 고르게 다시 나누는 Jumper 구조
# 덕분에 변화 뒤에도 빨리 이길 수 있어야 한다(first_alarm docstring).
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


# 확인: 자본 계산을 손으로 그대로 따라 한 반복문(재분배 → 베팅 함수 ε p^(ε−1) 곱하기 → 합이 1/δ 이상이면
# 경보)과 first_alarm의 경보 시점이 같다. δ를 1e-9로 올리면 네 장 안에는 닿지 않는다. 등록한 순서(재분배를
# 베팅 전에)를 구현이 지키는지 직접 대조한다.
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


# 확인: 섞을 결함 수 = round(r / (1 − r) × 정상 수)이고 0.5는 올리며(10.53 → 11), 그 범주의 결함 수(VisA test
# 40장)가 상한이다. M2AD 구간(정상 20장)에서는 5%·10%가 결함 1·2장이다. r = 1은 거부한다(7단계 규칙의 섞는
# 장수 정의).
def test_n_mixed():
    assert drift.n_mixed(200, 0.0, 40) == 0
    assert drift.n_mixed(200, 0.05, 40) == 11  # 10.53 rounds to 11
    assert drift.n_mixed(200, 0.10, 40) == 22
    assert drift.n_mixed(402, 0.10, 40) == 40  # capped by the defects there are
    assert drift.n_mixed(20, 0.05, 50) == 1 and drift.n_mixed(20, 0.10, 50) == 2
    with pytest.raises(ValueError):
        drift.n_mixed(10, 1.0, 5)


# 확인: 허위 경보 스트림은 정상 점수를 전부 한 번씩 담고 결함 k장을 더하며, 스트림마다 순서가 다르다(규칙:
# 범주마다 test 정상 전부를 무작위 순서로 늘어놓은 스트림).
def test_false_alarm_streams_hold_every_normal_once():
    rng = np.random.default_rng(3)
    normal, defect = np.arange(20.0), np.arange(100.0, 110.0)
    streams, k = drift.false_alarm_streams(normal, defect, 0.10, 5, rng)
    assert k == 2 and streams.shape == (5, 22)
    for row in streams:
        assert sorted(row[row < 100]) == list(normal) and (row >= 100).sum() == 2
    assert len({tuple(row) for row in streams}) == 5  # each stream has its own order


# 확인: VisA 합성 교란 지연 스트림은 정상과 결함을 반씩 나눠 앞 절반은 교란 없는 점수, 뒤 절반은 교란 점수로
# 놓고, 같은 이미지가 변화 앞뒤에 같이 나오지 않는다. 변화 위치는 앞 구간 길이(정상 5 + 결함 1)다.
# 왜: 같은 이미지가 양쪽에 있으면 점수가 짝지어 움직여 변화 감지가 실제보다 쉬워지거나 어려워질 수 있다(규칙:
# 같은 이미지가 양쪽에 나오지 않는다). clean과 changed에 1000 차이를 둬 어느 쪽 점수인지 값으로 가린다.
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


# 확인: M2AD 지연 스트림(I01 구간 뒤에 새 조명 구간)의 변화 위치와 섞인 결함 수, 그리고 지연 정의를 본다.
# 지연은 경보가 난 장을 포함해 변화 뒤 몇 장째인지(1 = 첫 장)이고, 변화 전 경보는 조기 경보로 따로 세며, 끝내
# 경보가 없으면 무한대다. pool_delays는 여러 검사기의 스트림을 합쳐 조기 경보 비율, 경보 비율, 중앙값을 내고,
# 중앙값이 무한대 쪽에 떨어지면 None이다(7단계 규칙의 경보 지연 정의).
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


# 확인: rng_for는 키 문자열의 SHA-256으로 시드를 정해 같은 키면 같은 난수가 나온다. detect는 같은 스트림·같은
# θ로 두 감지기를 돌리고, 100번째부터 점수가 크게 오른 스트림의 90% 넘게에서 둘 다 변화 뒤 30장 안에 경보한다.
# 보정 점수가 없으면(None) 전도형만 돈다.
# 왜: 두 감지기를 같은 무작위성으로 비교해야 차이가 방법에서만 나온다(규칙: 두 감지기는 같은 스트림과 같은 θ를
# 쓴다). 시드를 키로 정하는 것은 모듈 수준 난수를 쓰지 않고 결과를 결정적으로 만들기 위해서다.
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


# 저장된 VisA test 실행 폴더를 흉내 낸다. 방법마다 점수·보정 키 이름이 달라(PatchCore는 eval_score_full과 교차
# 적합 pool_score_oof, Dinomaly는 eval_score와 cal_score) VISA_SOURCES의 키를 그대로 받아 npz를 만든다. 범주
# 두 개, 정상 40장·결함 6장.
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


# 확인: 저장된 점수 파일만으로 세 표를 만든다.
# - fa_table: 섞임 비율 0/5/10%마다 두 감지기의 허위 경보 요약과 범주별 섞인 결함 수(정상 40장에 10%면 4장)를
#   낸다.
# - perturb_delay_table: 3-A perturb 실행의 clean 점수에서 교란 점수(+10)로 바뀌면 귀납형이 모든 스트림에서
#   경보하고 지연 중앙값이 5장 이하다.
# - m2ad_delay_table: 3-B 실행의 run.json과 검사기별 npz에서 기준 조명(S) 뒤의 새 조명(R:02)마다 표를 만든다.
#   라벨 −1(제외된 시편)은 정상·결함 어느 쪽에도 넣지 않는다.
# 왜: drift 명령은 이미지를 다시 읽지 않고 저장된 점수만 읽는다(GPU 없이 약 10분). 저장 형식의 키 이름이
# 바뀌면 이 표가 조용히 비거나 틀린 점수를 읽게 된다.
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
