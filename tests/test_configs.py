from dataclasses import asdict, replace

import pytest

from defect_inspect.configs import CONFIGS, get_config, same_config


def test_centring_configs_differ_from_their_base_only_in_centre():
    for base, centred in (("p0", "p0-c"), ("d-s", "d-s-c")):
        a, b = get_config(base), get_config(centred)
        assert not a.centre and b.centre
        assert replace(a, name=centred, centre=True) == b
    assert not any(cfg.centre for name, cfg in CONFIGS.items() if not name.endswith("-c"))


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


def test_unknown_config():
    with pytest.raises(ValueError):
        get_config("p1")
