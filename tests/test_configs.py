# [파일 역할] `defect_inspect.configs`의 고정 PatchCore 구성표(CONFIGS)와 `get_config`, `same_config`를
# 검증하는 테스트 묶음이다.
# [왜 필요한가] 구성값(백본, 입력 크기, 코어셋 비율 등)은 docs/experiments.md에 측정 전에 등록한 값이라 실수로
# 바뀌면 안 된다. 7단계(E1)에서 이미지별 특징 중심화 구성(p0-c, d-s-c)과 `centre` 필드를 더하면서, 기존 구성은
# 그대로이고 그 전에 기록된 실행도 여전히 같은 구성으로 인정되는지 확인할 필요가 생겼다.
# [연결] `same_config`는 run_perturb.py가 원본 실행 폴더의 기록된 구성이 요청한 등록 구성과 같은지 확인할 때
# 쓴다(다르면 parser.error로 멈춘다).

from dataclasses import asdict, replace

# asdict는 구성을 run.json에 기록되는 dict 모양으로 바꿀 때, replace는 '이름과 centre만 바꾼 사본'을 만들어
# 비교할 때 쓴다.
import pytest

from defect_inspect.configs import CONFIGS, get_config, same_config


# 확인: 중심화 구성(p0-c, d-s-c)이 바탕 구성(p0, d-s)과 이름과 centre만 다르고, '-c'로 끝나지 않는 구성은 모두
# centre가 꺼져 있다.
# 왜: 7단계 규칙은 E1을 '중심화 말고는 아무것도 바꾸지 않은' 구성으로 정의했다. 백본·크기·코어셋 비율이 같이
# 바뀌면 E1 − E0 차이를 중심화 효과로 읽을 수 없게 된다.
def test_centring_configs_differ_from_their_base_only_in_centre():
    for base, centred in (("p0", "p0-c"), ("d-s", "d-s-c")):
        a, b = get_config(base), get_config(centred)
        assert not a.centre and b.centre
        assert replace(a, name=centred, centre=True) == b
    assert not any(cfg.centre for name, cfg in CONFIGS.items() if not name.endswith("-c"))


# 확인: 7단계 이전 run.json(centre 키 없음)은 centre가 꺼진 구성으로 인정하고, 중심화 실행과 일반 구성은 서로
# 같다고 보지 않는다.
# 왜: 1~3단계 실행 폴더를 다시 쓰는 run_perturb가 centre 필드가 생겼다는 이유만으로 옛 실행을 거부하면 안
# 되고, 반대로 옛 기록이 중심화 구성으로 잘못 통과해서도 안 된다. sigma처럼 다른 값이 하나라도 다르거나 기록이
# dict가 아니면(None) 거부해야 한다.
def test_same_config_accepts_runs_recorded_before_the_centre_field():
    cfg = get_config("p0")
    recorded = asdict(cfg)
    assert same_config(recorded, cfg)
    old = {k: v for k, v in recorded.items() if k != "centre"}
    assert same_config(old, cfg)
    # A centred run is not the plain config, and an old record is never the centred one.
    assert not same_config(asdict(get_config("p0-c")), cfg)
    assert not same_config(old | {"name": "p0-c"}, get_config("p0-c"))
    assert same_config(asdict(get_config("p0-c")), get_config("p0-c"))
    assert not same_config(recorded | {"sigma": 2.0}, cfg)
    assert not same_config(None, cfg)


# 확인: 등록되지 않은 이름은 KeyError가 아니라 알려진 이름 목록을 담은 ValueError로 거부한다(CLI에서 오타를
# 바로 알 수 있게).
def test_unknown_config():
    with pytest.raises(ValueError):
        get_config("p1")
