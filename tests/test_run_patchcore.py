import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defect_inspect import analyze, run_patchcore  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.splits import ManifestRow, SealedTestError  # noqa: E402

SIZE = 32


class GridMean(torch.nn.Module):
    """Tiny stand-in for a backbone: the mean colour of each cell of a 4x4 grid."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 4).permute(0, 2, 3, 1)


class FakeCache:
    size = SIZE

    def __init__(self, manifest):
        rng = np.random.default_rng(0)
        self._images, self._masks = {}, {}
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255  # a bright square that no normal image has
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        return np.stack([self._masks[r.image] for r in rows])


def _manifest():
    rows = []
    for i in range(40):
        rows.append(ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, ""))
    for i in range(6):
        rows.append(ManifestRow(f"toy/d{i}.JPG", f"toy/d{i}.png", "toy", "anomaly", "dev_defect", -1, "hole"))
    for i in range(10):
        rows.append(ManifestRow(f"toy/tn{i}.JPG", "", "toy", "normal", "test_normal", -1, ""))
    for i in range(4):
        mask = f"toy/td{i}.png"
        rows.append(ManifestRow(f"toy/td{i}.JPG", mask, "toy", "anomaly", "test_defect", -1, "hole"))
    return rows


CFG = PatchCoreConfig(
    name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=0.5, sigma=1.0, batch_size=8
)


def _run(tmp_path, protocol, allow_test=False):
    manifest = _manifest()
    return run_patchcore.run_category(
        CFG, protocol, "toy", manifest, FakeCache(manifest), GridMean(), "cpu", allow_test, tmp_path
    )


def test_dev_run_writes_every_score_set(tmp_path):
    info = _run(tmp_path, "dev")
    assert (info["pool"], info["eval_normal"], info["eval_defect"]) == (32, 8, 6)
    assert info["grid"] == [4, 4] and info["dim"] == 3
    assert set(info["bank_rows"]) == {"full", "minus_fold_1", "minus_fold_2", "minus_fold_3", "minus_fold_4"}
    assert info["bank_rows"]["full"] == 32 * 16 // 2

    with np.load(tmp_path / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert set(z["pool_folds"].tolist()) == {1, 2, 3, 4}
        assert np.isfinite(z["pool_score_oof"]).all() and np.isfinite(z["pool_score_resub"]).all()
        # Defects (a bright square) score far above the normals under both models.
        for key in ("eval_score_full", "eval_score_holdout"):
            assert z[key][8:].min() > z[key][:8].max()
        # An image scored by a bank that contains its own patches looks more normal than when held out.
        assert z["pool_score_resub"].mean() < z["pool_score_oof"].mean()
        assert float(z["pixel_auroc"]) > 0.9
        assert z["pro_components"].shape[0] == 6
    maps = np.load(tmp_path / "toy_maps.npy")
    assert maps.shape == (14, 256, 256) and maps.dtype == np.float16


def test_dev_run_feeds_the_analysis(tmp_path, monkeypatch):
    info = _run(tmp_path, "dev")
    meta = {"config": {"name": "toy"}, "protocol": "dev", "commit": "test", "categories": [info]}
    (tmp_path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(analyze, "N_BOOT", 100)
    report = analyze.build_report(tmp_path)
    assert report["accuracy"]["macro_image_auroc"] == 1.0
    assert report["calibration"]["0.05"]["holdout"]["n_cal_total"] == 8
    assert report["calibration"]["0.05"]["crossfit"]["n_cal_total"] == 32


def test_test_protocol_needs_permission(tmp_path):
    with pytest.raises(SealedTestError):
        _run(tmp_path, "test")
    info = _run(tmp_path, "test", allow_test=True)
    assert (info["pool"], info["eval_normal"], info["eval_defect"]) == (40, 10, 4)
    assert "minus_fold_0" in info["bank_rows"]


def test_cli_refuses_test_protocol_without_flags(tmp_path):
    with pytest.raises(SystemExit):
        run_patchcore.main(["--protocol", "test", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        run_patchcore.main(["--protocol", "test", "--allow-test", "--out", str(tmp_path)])
