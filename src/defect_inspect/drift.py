"""Label-free change detection on a stream of anomaly scores (stage 7, reported without a verdict).

Each new image score becomes a conformal p-value: inductive (against a fixed set of calibration scores of
normal images) or transductive (against the scores the stream has shown so far, itself included). The
p-values feed a jumper test martingale with power betting functions; an alarm is raised when it reaches
1/DELTA. With exchangeable scores the transductive p-values are independent and uniform, so Ville's
inequality bounds the chance of any false alarm by DELTA over the whole stream. The inductive ones share
the calibration set and do not carry that guarantee.

`python -m defect_inspect.drift` reads saved scores only (no image, no GPU) and writes
reports/stage7/drift.json. Rules: docs/experiments.md, stage 7.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import paths
from .splits import path_key

EPSILONS = (1.0, 0.5, 0.2, 0.05)  # power betting eps * p**(eps - 1); eps = 1 is "no bet"
JUMP = 0.01
DELTA = 0.01
RATES = (0.0, 0.05, 0.10)  # share of defect images mixed into a stream
DETECTORS = ("inductive", "transductive")
N_FA_STREAMS = 1000
N_DELAY_STREAMS = 200
N_M2AD_STREAMS = 1000
_CHUNK = 100  # streams per block of the pairwise comparison in transductive_pvalues


def rng_for(key: str) -> np.random.Generator:
    """A generator seeded by the SHA-256 of `key`: every stream set has its own fixed seed."""
    return np.random.default_rng(int(path_key(key)[:16], 16))


def _theta(rng: np.random.Generator, shape) -> np.ndarray:
    """Uniform on (0, 1]: a p-value is never exactly 0."""
    return 1.0 - rng.random(shape)


def inductive_pvalues(scores: np.ndarray, cal: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """p = (#{cal > s} + theta * (#{cal = s} + 1)) / (n + 1), for every entry of `scores`."""
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    if cal.size == 0:
        raise ValueError("inductive p-values need calibration scores")
    s = np.asarray(scores, dtype=np.float64)
    right = np.searchsorted(cal, s, side="right")
    left = np.searchsorted(cal, s, side="left")
    return ((cal.size - right) + theta * (right - left + 1)) / (cal.size + 1)


def transductive_pvalues(scores: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """p_t = (#{i <= t: s_i > s_t} + theta_t * #{i <= t: s_i = s_t}) / t (1-based t), per stream (row)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"expected scores [streams, time], got {s.shape}")
    n_streams, length = s.shape
    past = np.tril(np.ones((length, length), dtype=bool))  # [t, i]: i <= t
    out = np.empty_like(s)
    for start in range(0, n_streams, _CHUNK):
        block = s[start : start + _CHUNK]
        later, earlier = block[:, :, None], block[:, None, :]
        greater = ((earlier > later) & past).sum(axis=2)
        equal = ((earlier == later) & past).sum(axis=2)
        out[start : start + _CHUNK] = (greater + theta[start : start + _CHUNK] * equal) / np.arange(
            1, length + 1
        )
    return out


def first_alarm(
    pvalues: np.ndarray, epsilons: Sequence[float] = EPSILONS, jump: float = JUMP, delta: float = DELTA
) -> np.ndarray:
    """Index of the first observation at which the jumper martingale reaches 1/delta (-1 if never).

    Capital starts split evenly over the betting functions eps * p**(eps - 1). Before each bet a share
    `jump` of the total is spread evenly again (Vovk et al. 2021, Simple Jumper), so a function that lost
    during a long quiet stretch can still win quickly after a change. Rows are independent streams.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"expected p-values [streams, time], got {p.shape}")
    if not ((p > 0) & (p <= 1)).all():
        raise ValueError("p-values must lie in (0, 1]")
    eps = np.asarray(epsilons, dtype=np.float64)[None, :]
    if not ((eps > 0) & (eps <= 1)).all():
        raise ValueError("every epsilon must lie in (0, 1]")
    k = eps.shape[1]
    capital = np.full((p.shape[0], k), 1.0 / k)
    alarm = np.full(p.shape[0], -1, dtype=np.int64)
    live = np.ones(p.shape[0], dtype=bool)
    threshold = 1.0 / delta
    for t in range(p.shape[1]):
        if not live.any():
            break
        c = capital[live]
        c = (1.0 - jump) * c + jump * c.sum(axis=1, keepdims=True) / k
        c = c * eps * p[live, t : t + 1] ** (eps - 1.0)
        capital[live] = c
        hit = np.flatnonzero(live)[c.sum(axis=1) >= threshold]
        alarm[hit] = t
        live[hit] = False  # only the first alarm counts; the capital of those rows is no longer updated
    return alarm


def n_mixed(n_normal: int, rate: float, available: int) -> int:
    """Defect images to mix into `n_normal` normal ones so that they make `rate` of the stream (capped)."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate must be in [0, 1), got {rate}")
    want = math.floor(rate / (1.0 - rate) * n_normal + 0.5)
    return min(want, int(available))


def _segment(rng: np.random.Generator, normal: np.ndarray, defect: np.ndarray, k: int) -> tuple:
    """All `normal` scores and k of the `defect` ones, in a random order: (scores, is_defect)."""
    chosen = rng.permutation(defect.size)[:k]
    scores = np.concatenate([normal, defect[chosen]])
    is_defect = np.concatenate([np.zeros(normal.size, bool), np.ones(k, bool)])
    order = rng.permutation(scores.size)
    return scores[order], is_defect[order]


def detect(scores: np.ndarray, cal: np.ndarray | None, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """First alarm of both detectors on the same streams and the same randomisation."""
    theta = _theta(rng, scores.shape)
    out = {"transductive": first_alarm(transductive_pvalues(scores, theta))}
    if cal is not None:
        out["inductive"] = first_alarm(inductive_pvalues(scores, cal, theta))
    return out


def false_alarm_streams(
    normal: np.ndarray, defect: np.ndarray, rate: float, n_streams: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """[n_streams, T]: every normal score once plus k defect scores, each stream in its own random order."""
    normal, defect = np.asarray(normal, np.float64), np.asarray(defect, np.float64)
    k = n_mixed(normal.size, rate, defect.size)
    rows = [_segment(rng, normal, defect, k)[0] for _ in range(n_streams)]
    return np.stack(rows), k


def change_streams(
    pre: tuple[np.ndarray, np.ndarray],
    post: tuple[np.ndarray, np.ndarray],
    rate: float,
    n_streams: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, tuple[int, int]]:
    """[n_streams, T]: a pre-change segment then a post-change one, each (normals, defects) shuffled.

    Returns the streams, the index of the first post-change observation, and the defects mixed (pre, post).
    """
    (pre_n, pre_d), (post_n, post_d) = [
        (np.asarray(a, np.float64), np.asarray(b, np.float64)) for a, b in (pre, post)
    ]
    k_pre = n_mixed(pre_n.size, rate, pre_d.size)
    k_post = n_mixed(post_n.size, rate, post_d.size)
    rows = [
        np.concatenate([_segment(rng, pre_n, pre_d, k_pre)[0], _segment(rng, post_n, post_d, k_post)[0]])
        for _ in range(n_streams)
    ]
    return np.stack(rows), pre_n.size + k_pre, (k_pre, k_post)


def split_change_streams(
    clean: np.ndarray,
    changed: np.ndarray,
    labels: np.ndarray,
    rate: float,
    n_streams: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Streams from one image set: per stream a random half of the normals (and of the defects) comes
    before the change with its clean scores, the other half after it with its changed scores.

    No image appears in both segments. Returns the streams and the index of the first post-change one.
    """
    clean, changed = np.asarray(clean, np.float64), np.asarray(changed, np.float64)
    normal, defect = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    half_n, half_d = normal.size // 2, defect.size // 2
    k_pre = n_mixed(half_n, rate, half_d)
    k_post = n_mixed(normal.size - half_n, rate, defect.size - half_d)
    rows = []
    for _ in range(n_streams):
        n_perm, d_perm = rng.permutation(normal), rng.permutation(defect)
        pre = _segment(rng, clean[n_perm[:half_n]], clean[d_perm[:half_d]], k_pre)[0]
        post = _segment(rng, changed[n_perm[half_n:]], changed[d_perm[half_d:]], k_post)[0]
        rows.append(np.concatenate([pre, post]))
    return np.stack(rows), half_n + k_pre


def summarise_fa(alarm: np.ndarray) -> dict:
    return {"streams": int(alarm.size), "alarms": int((alarm >= 0).sum()), "rate": float((alarm >= 0).mean())}


def delays(alarm: np.ndarray, change: int) -> tuple[int, np.ndarray]:
    """(early alarms, delays of the other streams): the delay counts post-change images up to and
    including the alarm (1 = the first one); a stream without an alarm after the change gets inf."""
    alarm = np.asarray(alarm)
    early = (alarm >= 0) & (alarm < change)
    rest = alarm[~early]
    return int(early.sum()), np.where(rest >= change, rest - change + 1, np.inf).astype(np.float64)


def pool_delays(parts: Sequence[tuple[np.ndarray, int]]) -> dict:
    """Delay summary over the streams of several categories or inspectors, each with its change index.

    The median and the 90th percentile count streams without an alarm as infinitely late; they are None
    when they fall on such a stream.
    """
    early, rest = 0, []
    for alarm, change in parts:
        e, d = delays(alarm, change)
        early += e
        rest.append(d)
    late = np.concatenate(rest) if rest else np.zeros(0)
    streams = early + late.size

    def finite_or_none(x: float) -> float | None:
        return float(x) if math.isfinite(x) else None

    return {
        "streams": streams,
        "early_share": early / streams if streams else float("nan"),
        "alarmed_share": float(np.isfinite(late).mean()) if late.size else float("nan"),
        "median_delay": finite_or_none(np.median(late)) if late.size else None,
        "p90_delay": finite_or_none(np.percentile(late, 90, method="higher")) if late.size else None,
    }


# ---------------------------------------------------------------- saved scores


VISA_SOURCES = {
    # method: (run directory, test score key, calibration score key)
    "p0": ("p0-test", "eval_score_full", "pool_score_oof"),
    "d-s": ("d-s-test", "eval_score_full", "pool_score_oof"),
    "dm": ("dm-test", "eval_score", "cal_score"),
    "dms-280-car": ("dms-280-car-onnx-fp32-test", "eval_score", "cal_score"),
}
PERTURB_METHODS = ("p0", "d-s", "dm")
M2AD_METHODS = ("p0", "d-s")


def load_visa(method: str, outputs: Path) -> dict[str, dict[str, np.ndarray]]:
    run, score_key, cal_key = VISA_SOURCES[method]
    out = {}
    for path in sorted((Path(outputs) / run).glob("*.npz")):
        with np.load(path) as z:
            if score_key not in z.files:
                continue
            out[path.stem] = {"scores": z[score_key], "labels": z["eval_labels"], "cal": z[cal_key]}
    if not out:
        raise FileNotFoundError(f"no saved {method} test scores in {Path(outputs) / run}")
    return out


def fa_table(method: str, outputs: Path, n_streams: int = N_FA_STREAMS) -> dict:
    """False alarms on streams of a method's saved VisA test scores, per detector and defect share."""
    cats = load_visa(method, outputs)
    table: dict = {}
    for rate in RATES:
        alarms = {d: [] for d in DETECTORS}
        per_category: dict = {}
        mixed = {}
        for category, arr in cats.items():
            labels = arr["labels"]
            rng = rng_for(f"fa|{method}|{category}|{rate}")
            streams, k = false_alarm_streams(
                arr["scores"][labels == 0], arr["scores"][labels == 1], rate, n_streams, rng
            )
            mixed[category] = k
            found = detect(streams, arr["cal"], rng)
            per_category[category] = {d: summarise_fa(found[d])["rate"] for d in DETECTORS}
            for d in DETECTORS:
                alarms[d].append(found[d])
        table[f"{rate:.2f}"] = {d: summarise_fa(np.concatenate(alarms[d])) for d in DETECTORS} | {
            "mixed_defects": mixed,
            "per_category": per_category,
        }
    return table


def perturb_delay_table(method: str, outputs: Path, n_streams: int = N_DELAY_STREAMS) -> dict:
    """Alarm delay after a switch from clean scores to the scores of each synthetic condition (VisA test)."""
    run = Path(outputs) / f"perturb-{method}-test"
    cats = {}
    for path in sorted(run.glob("*.npz")):
        with np.load(path) as z:
            cats[path.stem] = {k: z[k] for k in ("conditions", "scores", "eval_labels", "cal_score")}
    if not cats:
        raise FileNotFoundError(f"no perturbation run in {run}")
    conditions = [str(c) for c in next(iter(cats.values()))["conditions"]]
    table: dict = {}
    for condition in conditions[1:]:
        table[condition] = {}
        for rate in RATES:
            pooled = {d: [] for d in DETECTORS}
            for category, arr in cats.items():
                names = [str(c) for c in arr["conditions"]]
                rng = rng_for(f"visa-delay|{method}|{category}|{condition}|{rate}")
                streams, change = split_change_streams(
                    arr["scores"][names.index("clean")],
                    arr["scores"][names.index(condition)],
                    arr["eval_labels"],
                    rate,
                    n_streams,
                    rng,
                )
                found = detect(streams, arr["cal_score"], rng)
                for d in DETECTORS:
                    pooled[d].append((found[d], change))
            table[condition][f"{rate:.2f}"] = {d: pool_delays(pooled[d]) for d in DETECTORS}
    return table


def m2ad_delay_table(method: str, outputs: Path, n_streams: int = N_M2AD_STREAMS) -> dict:
    """Alarm delay after a switch from I01 to each other real illumination (stage 3-B saved scores)."""
    run = Path(outputs) / f"m2ad-{method}"
    with open(run / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    inspectors = {}
    for info in meta["inspectors"]:
        name = f"{info['category']}_{info['view']}"
        with np.load(run / f"{name}.npz") as z:
            inspectors[name] = {k: z[k] for k in ("conditions", "scores", "labels", "cal_score")}
    lights = [
        c.split(":")[1] for c in next(iter(inspectors.values()))["conditions"].tolist() if c.startswith("R:")
    ]
    table: dict = {}
    for light in lights:
        table[light] = {}
        for rate in RATES:
            pooled = {d: [] for d in DETECTORS}
            for name, arr in inspectors.items():
                conditions = arr["conditions"].tolist()
                s_ref, s_new = conditions.index("S"), conditions.index(f"R:{light}")
                pre = (
                    arr["scores"][s_ref][arr["labels"][s_ref] == 0],
                    arr["scores"][s_ref][arr["labels"][s_ref] == 1],
                )
                post = (
                    arr["scores"][s_new][arr["labels"][s_new] == 0],
                    arr["scores"][s_new][arr["labels"][s_new] == 1],
                )
                rng = rng_for(f"m2ad-delay|{method}|{name}|{light}|{rate}")
                streams, change, _ = change_streams(pre, post, rate, n_streams, rng)
                found = detect(streams, arr["cal_score"], rng)
                for d in DETECTORS:
                    pooled[d].append((found[d], change))
            table[light][f"{rate:.2f}"] = {d: pool_delays(pooled[d]) for d in DETECTORS}
    return table


def build_report(
    outputs: Path,
    fa_streams: int = N_FA_STREAMS,
    delay_streams: int = N_DELAY_STREAMS,
    m2ad_streams: int = N_M2AD_STREAMS,
) -> dict:
    return {
        "settings": {
            "epsilons": list(EPSILONS),
            "jump": JUMP,
            "delta": DELTA,
            "rates": list(RATES),
            "fa_streams_per_category": fa_streams,
            "delay_streams_per_category": delay_streams,
            "m2ad_streams_per_inspector": m2ad_streams,
        },
        "false_alarms": {m: fa_table(m, outputs, fa_streams) for m in VISA_SOURCES},
        "visa_delay": {m: perturb_delay_table(m, outputs, delay_streams) for m in PERTURB_METHODS},
        "m2ad_delay": {m: m2ad_delay_table(m, outputs, m2ad_streams) for m in M2AD_METHODS},
    }


def format_report(report: dict) -> str:
    lines = ["false alarms on VisA test normals (share of streams with any alarm):"]
    for method, table in report["false_alarms"].items():
        cells = [
            f"{rate}: ind {row['inductive']['rate'] * 100:.2f}% / "
            f"trans {row['transductive']['rate'] * 100:.2f}%"
            for rate, row in table.items()
        ]
        lines.append(f"  {method}: " + "; ".join(cells))
    for key, title in (("m2ad_delay", "M2AD illuminations"), ("visa_delay", "VisA conditions")):
        lines.append(f"median delay after the change ({title}, rate 0.00, ind / trans):")
        for method, table in report[key].items():
            cells = [
                f"{cond} {row['0.00']['inductive']['median_delay']}/"
                f"{row['0.00']['transductive']['median_delay']}"
                for cond, row in table.items()
            ]
            lines.append(f"  {method}: " + ", ".join(cells))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = build_report(paths.OUTPUTS)
    print(format_report(report), end="")
    if not args.no_write:
        from .analyze_m2ad import strict_json

        out = paths.REPORTS / "stage7" / "drift.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(strict_json(report), f, ensure_ascii=False, indent=1, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
