"""CPU runtimes for one reconstruction artifact: onnxruntime against OpenVINO (score parity, then latency).

Arms (docs/experiments.md, "CPU 런타임 대조", registered 2026-10-04 before measuring):

- R0: onnxruntime CPUExecutionProvider, intra-op threads at its default (the serving setting)
- R1: onnxruntime, 12 intra-op threads (control)
- R2: OpenVINO on the CPU, reading the same ONNX file: PERFORMANCE_HINT=LATENCY,
  INFERENCE_PRECISION_HINT=f32, NUM_STREAMS=1

`parity` scores the dev evaluation set (fold-0 normals + dev defects) image by image with every arm and
compares scores, maps, AUROC and the hold-out flags with R0 (and R0 with the recorded dev run).
`latency` waits for an idle machine, then times the arms in blocks whose order rotates, each run in a
fresh process (`run-arm`). Neither command reads the sealed test set. OpenVINO is imported only when an
OpenVINO arm is built (dependency group `openvino`).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import conformal_threshold
from .download import sha256_file
from .inspector import RECONSTRUCTION_PRECISIONS, ReconstructionInspector, check_reconstruction_meta
from .ledger import git_commit
from .metrics import auroc

ARMS: dict[str, dict] = {
    "R0": {"runtime": "onnxruntime", "threads": None},
    "R1": {"runtime": "onnxruntime", "threads": 12},
    "R2": {
        "runtime": "openvino",
        "config": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "1"},
    },
}
REFERENCE = "R0"
ALPHA = 0.05
SCORE_REL_TOL = 1e-3  # (a) largest relative image-score difference against R0
MAP_ABS_TOL = 1e-3  # (b) largest absolute map difference against R0
AUROC_TOL_PP = 0.01  # (c) dev macro image AUROC, percentage points
RECORD_REL_TOL = 1e-6  # R0 against the recorded dev run (same file, same runtime)
MIN_SPEEDUP = 0.10  # H22
CPU_IDLE_PCT = 10.0
GPU_IDLE_PCT = 10.0
OV_PROPERTIES = (
    "PERFORMANCE_HINT",
    "INFERENCE_PRECISION_HINT",
    "NUM_STREAMS",
    "INFERENCE_NUM_THREADS",
    "ENABLE_HYPER_THREADING",
    "ENABLE_CPU_PINNING",
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- arms


class OpenVINOSession:
    """The part of an onnxruntime session that `ReconstructionInspector` uses, backed by OpenVINO.

    The ONNX file is read directly (no converted IR). One infer request is reused, like one session.
    """

    def __init__(self, model_path: Path, config: dict, device: str = "CPU"):
        import openvino as ov

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(model_path)), device, dict(config))
        self.request = self.compiled.create_infer_request()
        self.version = ov.__version__
        self.properties = {}
        for name in OV_PROPERTIES:
            try:
                self.properties[name] = str(self.compiled.get_property(name))
            except Exception:  # a property this plugin does not report
                self.properties[name] = None

    def run(self, names: list[str], feeds: dict) -> list[np.ndarray]:
        result = self.request.infer(feeds)
        return [np.array(result[self.compiled.output(name)]) for name in names]


def _model_path(category_dir: Path, precision: str = "fp32") -> Path:
    """The model file of a category folder; like `ReconstructionInspector.load`, it may sit in the parent."""
    name = RECONSTRUCTION_PRECISIONS[precision]
    path = Path(category_dir) / name
    return path if path.exists() else Path(category_dir).parent / name


def load_arm(arm: str, category_dir: Path) -> ReconstructionInspector:
    """The FP32 inspector of one category under one arm."""
    spec = ARMS[arm]
    if spec["runtime"] == "onnxruntime":
        return ReconstructionInspector.load(category_dir, precision="fp32", threads=spec["threads"])
    with open(Path(category_dir) / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    check_reconstruction_meta(meta)
    inspector = ReconstructionInspector(OpenVINOSession(_model_path(category_dir), spec["config"]), meta)
    inspector.precision = "fp32"
    return inspector


def arm_info(arm: str, inspector: ReconstructionInspector) -> dict:
    spec = ARMS[arm]
    info: dict = {"runtime": spec["runtime"]}
    if spec["runtime"] == "onnxruntime":
        import onnxruntime as ort

        info.update(version=ort.__version__, intra_op_threads=spec["threads"])
    else:
        info.update(version=inspector.session.version, config=spec["config"])
        info["compiled"] = inspector.session.properties
    return info


# ---------------------------------------------------------------- parity


def max_rel_diff(values: np.ndarray, ref: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values - ref) / np.maximum(np.abs(ref), 1e-12)))


def category_parity(ref: np.ndarray, arm: np.ndarray, labels: np.ndarray, stored_threshold: float) -> dict:
    """One category: an arm's image scores against R0's on the same images (labels 0 = fold-0 normal)."""
    ref = np.asarray(ref, dtype=np.float32)
    arm = np.asarray(arm, dtype=np.float32)
    labels = np.asarray(labels)
    normal = labels == 0
    thr_ref = conformal_threshold(ref[normal], ALPHA)
    thr_arm = conformal_threshold(arm[normal], ALPHA)
    flags_ref = ref > thr_ref.value
    flags_arm = arm > thr_arm.value
    return {
        "images": int(labels.size),
        "normal": int(normal.sum()),
        "defect": int((~normal).sum()),
        "max_rel_diff": max_rel_diff(arm, ref),
        "auroc_ref": auroc(ref[normal], ref[~normal]),
        "auroc_arm": auroc(arm[normal], arm[~normal]),
        "threshold_ref": thr_ref.value,
        "threshold_arm": thr_arm.value,
        "threshold_rank": thr_ref.rank,
        # (d): the arm's own hold-out threshold flags the same fold-0 normals as R0's own.
        "fold0_flags_ref": int(flags_ref[normal].sum()),
        "fold0_flags_same": bool(np.array_equal(flags_ref[normal], flags_arm[normal])),
        # Reported only: defects under the own thresholds, all images under the stored threshold.
        "defect_verdict_flips_own": int((flags_ref[~normal] != flags_arm[~normal]).sum()),
        "verdict_flips_stored": int(((ref > stored_threshold) != (arm > stored_threshold)).sum()),
    }


def parity_gate(categories: dict[str, dict], map_abs: float) -> dict:
    """Aggregate the per-category results of one arm into the registered gate (a)-(d)."""
    rel = max(c["max_rel_diff"] for c in categories.values())
    auroc_ref = 100.0 * float(np.mean([c["auroc_ref"] for c in categories.values()]))
    auroc_arm = 100.0 * float(np.mean([c["auroc_arm"] for c in categories.values()]))
    checks = {
        "a_score_rel": {"value": rel, "limit": SCORE_REL_TOL, "pass": rel <= SCORE_REL_TOL},
        "b_map_abs": {"value": map_abs, "limit": MAP_ABS_TOL, "pass": map_abs <= MAP_ABS_TOL},
        "c_macro_auroc_pp": {
            "value": auroc_arm - auroc_ref,
            "arm": auroc_arm,
            "ref": auroc_ref,
            "limit": AUROC_TOL_PP,
            "pass": abs(auroc_arm - auroc_ref) <= AUROC_TOL_PP,
        },
        "d_fold0_flags": {
            "same_in": sum(c["fold0_flags_same"] for c in categories.values()),
            "categories": len(categories),
            "pass": all(c["fold0_flags_same"] for c in categories.values()),
        },
    }
    return {
        "checks": checks,
        "pass": all(v["pass"] for v in checks.values()),
        "defect_verdict_flips_own": sum(c["defect_verdict_flips_own"] for c in categories.values()),
        "verdict_flips_stored": sum(c["verdict_flips_stored"] for c in categories.values()),
    }


def _parity_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from .cache import ImageCache
    from .splits import read_manifest, select

    art = args.artifacts
    arms = args.arms
    if arms[0] != REFERENCE or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        parser.error(f"arms must start with {REFERENCE} and be distinct names from {sorted(ARMS)}")
    started = time.perf_counter()
    manifest = read_manifest(paths.VISA_MANIFEST)
    model_path = art / RECONSTRUCTION_PRECISIONS["fp32"]
    with open(args.recorded / "run.json", encoding="utf-8") as f:
        recorded_run = json.load(f)
    size = None
    errors: dict[str, str] = {}
    per_arm: dict[str, dict] = {a: {} for a in arms}
    map_abs = {a: 0.0 for a in arms if a != REFERENCE}
    record_rel = 0.0
    seconds = {a: 0.0 for a in arms}
    info: dict[str, dict] = {}
    for category in args.categories:
        inspectors = {}
        for arm in arms:
            if arm in errors:
                continue
            try:
                inspectors[arm] = load_arm(arm, art / category)
            except Exception as err:  # an arm that cannot load this model fails its gate
                if arm == REFERENCE:
                    raise
                errors[arm] = f"{type(err).__name__}: {err}"
                print(json.dumps({"arm": arm, "error": errors[arm]}), flush=True)
        for arm, insp in inspectors.items():
            info.setdefault(arm, arm_info(arm, insp))
        size = size or inspectors[REFERENCE].size
        normal = select(manifest, protocol="dev", part="eval_normal", category=category)
        defect = select(manifest, protocol="dev", part="eval_defect", category=category)
        rows = normal + defect
        labels = np.array([0] * len(normal) + [1] * len(defect), dtype=np.int8)
        with np.load(args.recorded / f"{category}.npz") as z:
            if z["eval_images"].tolist() != [r.image for r in rows]:
                raise RuntimeError(f"{args.recorded}/{category}.npz lists other images")
            recorded = z["eval_score"].astype(np.float32)
        images = ImageCache(paths.CACHE, size).images(rows)
        scores = {arm: np.empty(len(rows), dtype=np.float32) for arm in inspectors}
        for i, image in enumerate(images):
            maps = {}
            for arm, insp in inspectors.items():
                t0 = time.perf_counter()
                scores[arm][i], maps[arm] = insp.run(image)
                seconds[arm] += time.perf_counter() - t0
            for arm in maps:
                if arm != REFERENCE:
                    diff = float(np.max(np.abs(maps[arm] - maps[REFERENCE])))
                    map_abs[arm] = max(map_abs[arm], diff)
        record_rel = max(record_rel, max_rel_diff(scores[REFERENCE], recorded))
        stored = inspectors[REFERENCE].threshold
        for arm in inspectors:
            per_arm[arm][category] = category_parity(scores[REFERENCE], scores[arm], labels, stored)
        line = {a: round(per_arm[a][category]["max_rel_diff"], 9) for a in inspectors if a != REFERENCE}
        print(json.dumps({"category": category, "images": len(rows), "max_rel_diff": line}), flush=True)

    arms_report = {}
    for arm in arms:
        if arm == REFERENCE:
            continue
        if arm in errors:
            arms_report[arm] = {"pass": False, "error": errors[arm]}
            continue
        arms_report[arm] = {
            **info[arm],
            **parity_gate(per_arm[arm], map_abs[arm]),
            "score_seconds": round(seconds[arm], 1),
            "categories": per_arm[arm],
        }
    report = {
        "check": "CPU runtime parity on the dev evaluation set (docs/experiments.md, 2026-10-04)",
        "commit": git_commit(),
        "artifacts": _rel(art),
        "onnx_sha256": sha256_file(model_path),
        "reference": {
            **info[REFERENCE],
            "arm": REFERENCE,
            "score_seconds": round(seconds[REFERENCE], 1),
            "recorded_run": _rel(args.recorded),
            "recorded_onnx_sha256": recorded_run["config"]["onnx_sha256"],
            "max_rel_diff_vs_record": record_rel,
            "matches_record": record_rel <= RECORD_REL_TOL,
        },
        "images": sum(c["images"] for c in per_arm[REFERENCE].values()),
        "tolerances": {
            "score_rel": SCORE_REL_TOL,
            "map_abs": MAP_ABS_TOL,
            "auroc_pp": AUROC_TOL_PP,
            "record_rel": RECORD_REL_TOL,
            "alpha": ALPHA,
        },
        "arms": arms_report,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    _write_json(args.out, report)
    summary = {a: v.get("pass") for a, v in arms_report.items()}
    print(json.dumps({"matches_record": report["reference"]["matches_record"], "pass": summary}), flush=True)


# ---------------------------------------------------------------- machine load


def _filetime(ft) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def _system_times() -> tuple[int, int] | None:
    """(idle, kernel + user) in 100 ns units on Windows; kernel time includes idle time."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
    return _filetime(idle), _filetime(kernel) + _filetime(user)


def busy_percent(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    if before is None or after is None or after[1] <= before[1]:
        return None
    return 100.0 * (1.0 - (after[0] - before[0]) / (after[1] - before[1]))


def cpu_percent(seconds: float) -> float | None:
    """Mean total CPU utilisation over the next `seconds` (None where it cannot be read)."""
    before = _system_times()
    time.sleep(seconds)
    return busy_percent(before, _system_times())


def gpu_percent() -> float | None:
    """GPU utilisation from nvidia-smi (the first GPU), or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def load_minute(step_s: float = 5.0, steps: int = 12) -> dict:
    """CPU utilisation over one minute and the mean of GPU samples taken every `step_s` seconds."""
    before = _system_times()
    gpu = []
    for _ in range(steps):
        time.sleep(step_s)
        value = gpu_percent()
        if value is not None:
            gpu.append(value)
    return {
        "cpu_pct": busy_percent(before, _system_times()),
        "gpu_pct": float(np.mean(gpu)) if gpu else None,
        "gpu_samples": len(gpu),
    }


def is_idle(minute: dict, cpu_limit: float = CPU_IDLE_PCT, gpu_limit: float = GPU_IDLE_PCT) -> bool:
    cpu, gpu = minute.get("cpu_pct"), minute.get("gpu_pct")
    return cpu is not None and cpu < cpu_limit and (gpu is None or gpu < gpu_limit)


def wait_idle(max_minutes: int, minute=load_minute, clock=time.strftime) -> tuple[bool, list[dict]]:
    """Measure one minute at a time until the machine is idle or `max_minutes` have passed."""
    log = []
    for i in range(max_minutes):
        load = minute()  # measure first: `end` is the time the minute ended (2026-10-04 fix)
        record = {"minute": i + 1, "end": clock("%Y-%m-%d %H:%M:%S"), **load}
        record["idle"] = is_idle(record)
        log.append(record)
        print(json.dumps(record), flush=True)
        if record["idle"]:
            return True, log
    return False, log


# ---------------------------------------------------------------- latency


def block_orders(arms: list[str], blocks: int) -> list[list[str]]:
    """Block b runs the arms rotated by b, so every arm takes every position once in len(arms) blocks."""
    return [arms[b % len(arms) :] + arms[: b % len(arms)] for b in range(blocks)]


def time_calls(inspector, images: np.ndarray, warmup: int, repeats: int) -> list[float]:
    """Milliseconds of each timed `inspector.run` call (as `bench.time_reconstruction`, raw values)."""
    if len(images) == 0 or warmup < 0 or repeats < 1:
        raise ValueError("need images, warmup >= 0 and repeats >= 1")
    out = []
    for i in range(warmup + repeats):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        inspector.run(image)
        t1 = time.perf_counter()
        if i >= warmup:
            out.append((t1 - t0) * 1e3)
    return out


def _latency_images(category: str, size: int, n: int) -> np.ndarray:
    from .cache import ImageCache
    from .splits import read_manifest, select

    rows = select(read_manifest(paths.VISA_MANIFEST), protocol="dev", part="pool_normal", category=category)
    return ImageCache(paths.CACHE, size).images(rows[:n])


def _run_arm_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    t0 = time.perf_counter()
    inspector = load_arm(args.arm, args.artifacts / args.category)
    load_s = time.perf_counter() - t0
    images = _latency_images(args.category, inspector.size, args.images)
    ms = time_calls(inspector, images, args.warmup, args.repeats)
    print(
        json.dumps({"arm": args.arm, "load_s": load_s, "ms": ms, **arm_info(args.arm, inspector)}), flush=True
    )


def summarise_latency(runs: list[dict], resize_ms: float, arms: list[str]) -> dict:
    """Per arm: the median of all timed calls of all blocks plus the resize, and the gain against R0."""
    out: dict = {}
    for arm in arms:
        mine = [r for r in runs if r["arm"] == arm]
        pooled = [v for r in mine for v in r["ms"]]
        median = float(statistics.median(pooled))
        out[arm] = {
            "inference_median_ms": median,
            "inference_p95_ms": float(np.percentile(pooled, 95)),
            "block_medians_ms": [float(statistics.median(r["ms"])) for r in mine],
            "timed_calls": len(pooled),
            "total_with_resize_ms": resize_ms + median,
        }
    ref = out[REFERENCE]["total_with_resize_ms"]
    for arm in arms:
        out[arm]["speedup_vs_r0"] = 1.0 - out[arm]["total_with_resize_ms"] / ref
        out[arm]["inference_speedup_vs_r0"] = (
            1.0 - out[arm]["inference_median_ms"] / out[REFERENCE]["inference_median_ms"]
        )
    return out


def h22(parity: dict | None, latency: dict) -> dict:
    """H22 per challenger arm: passes the parity gate and is at least 10 % faster than R0 in this session.

    The parity gate only counts when R0 itself reproduced the recorded dev scores.
    """
    out = {}
    for arm, lat in latency.items():
        if arm == REFERENCE:
            continue
        passed = None
        if parity is not None:
            reference_ok = bool(parity.get("reference", {}).get("matches_record", False))
            passed = reference_ok and bool(parity.get("arms", {}).get(arm, {}).get("pass", False))
        faster = lat["speedup_vs_r0"] >= MIN_SPEEDUP
        out[arm] = {
            "parity_pass": passed,
            "speedup_vs_r0": lat["speedup_vs_r0"],
            "faster_by_10pct": faster,
            "candidate": bool(passed and faster),
        }
    return out


def _run_arm_cmd(args: argparse.Namespace, arm: str) -> list[str]:
    options = {
        "--arm": arm,
        "--artifacts": str(args.artifacts),
        "--category": args.category,
        "--images": str(args.images),
        "--warmup": str(args.warmup),
        "--repeats": str(args.repeats),
    }
    cmd = [sys.executable, "-m", "defect_inspect.runtime_compare", "run-arm"]
    for name, value in options.items():
        cmd += [name, value]
    return cmd


def _rel(path: Path) -> str:
    """A path relative to the repository when it lies inside it (reports carry no local prefixes)."""
    try:
        return Path(path).resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        return str(path)


def _latency_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from .bench import cpu_name, time_resize

    arms = args.arms
    if arms[0] != REFERENCE or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        parser.error(f"arms must start with {REFERENCE} and be distinct names from {sorted(ARMS)}")
    with open(args.artifacts / args.category / "meta.json", encoding="utf-8") as f:
        size = int(json.load(f)["img_size"])
    idle, wait_log = wait_idle(args.wait_max_min)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    resize_ms = time_resize(size)
    runs = []
    for b, order in enumerate(block_orders(arms, args.blocks)):
        for arm in order:
            cpu5, gpu = cpu_percent(5.0), gpu_percent()
            done = subprocess.run(_run_arm_cmd(args, arm), capture_output=True, text=True, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"run-arm {arm} failed: {done.stderr[-2000:]}")
            run = json.loads(done.stdout.strip().splitlines()[-1])
            run.update(block=b + 1, cpu5_before_pct=cpu5, gpu_before_pct=gpu)
            runs.append(run)
            print(
                json.dumps(
                    {k: v for k, v in run.items() if k != "ms"} | {"median_ms": statistics.median(run["ms"])}
                ),
                flush=True,
            )
    after = load_minute()
    summary = summarise_latency(runs, resize_ms, arms)
    parity = None
    if args.parity is not None and args.parity.exists():
        with open(args.parity, encoding="utf-8") as f:
            parity = json.load(f)
    report = {
        "check": "CPU runtime latency (docs/experiments.md, 2026-10-04)",
        "commit": git_commit(),
        "artifacts": _rel(args.artifacts),
        "onnx_sha256": sha256_file(args.artifacts / RECONSTRUCTION_PRECISIONS["fp32"]),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "category": args.category,
        "images": args.images,
        "warmup": args.warmup,
        "repeats_per_run": args.repeats,
        "blocks": block_orders(arms, args.blocks),
        "idle": {
            "reached": idle,
            "cpu_limit_pct": CPU_IDLE_PCT,
            "gpu_limit_pct": GPU_IDLE_PCT,
            "wait": wait_log,
        },
        "started": started,
        "after_minute": after,
        "resize_ms": resize_ms,
        "arms": summary,
        "H22": h22(parity, summary),
        "parity_report": None if parity is None else _rel(args.parity),
        "runs": runs,
    }
    _write_json(args.out, report)
    print(json.dumps({"arms": summary, "H22": report["H22"]}, indent=1), flush=True)


def main(argv: list[str] | None = None) -> None:
    from .visa import CATEGORIES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    default_art = paths.ROOT / "artifacts" / "dms-280-car"

    par = commands.add_parser("parity", help="score the dev evaluation set with every arm and compare")
    par.add_argument("--artifacts", type=Path, default=default_art)
    par.add_argument("--recorded", type=Path, default=paths.OUTPUTS / "dms-280-car-onnx-fp32-dev")
    par.add_argument("--arms", nargs="+", default=list(ARMS))
    par.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    par.add_argument("--out", type=Path, default=paths.REPORTS / "runtime" / "parity.json")

    lat = commands.add_parser("latency", help="time the arms in rotating blocks on an idle machine")
    lat.add_argument("--artifacts", type=Path, default=default_art)
    lat.add_argument("--arms", nargs="+", default=list(ARMS))
    lat.add_argument("--category", default="pcb1")
    lat.add_argument("--images", type=int, default=50)
    lat.add_argument("--warmup", type=int, default=5)
    lat.add_argument("--repeats", type=int, default=50)
    lat.add_argument("--blocks", type=int, default=3)
    lat.add_argument("--wait-max-min", type=int, default=60, help="minutes to wait for an idle machine")
    lat.add_argument("--parity", type=Path, default=paths.REPORTS / "runtime" / "parity.json")
    lat.add_argument("--out", type=Path, default=paths.REPORTS / "runtime" / "latency.json")

    one = commands.add_parser("run-arm", help="(used by latency) time one arm in this process")
    one.add_argument("--arm", choices=sorted(ARMS), required=True)
    one.add_argument("--artifacts", type=Path, required=True)
    one.add_argument("--category", default="pcb1")
    one.add_argument("--images", type=int, default=50)
    one.add_argument("--warmup", type=int, default=5)
    one.add_argument("--repeats", type=int, default=50)

    args = parser.parse_args(argv)
    {"parity": _parity_command, "latency": _latency_command, "run-arm": _run_arm_command}[args.command](
        args, parser
    )


if __name__ == "__main__":
    main()
