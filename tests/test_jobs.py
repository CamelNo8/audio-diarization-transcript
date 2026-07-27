"""未知話者ラベリングのジョブ管理と CSV 再ラベルの特性テスト。

対象は :mod:`src.web.jobs` の公開関数。
job_id のパストラバーサル検査（規約7.1）もここで固定する。
"""

from __future__ import annotations

import csv
import json
import re

import pytest

import src.web.jobs as jobs


@pytest.fixture(autouse=True)
def _保存先を差し替える(ジョブ保存先):
    """全テストで保存先とメモリキャッシュを tmp_path 配下に固定する（conftest）。"""


def _write_csv(path, rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows(rows)


def _read_csv(path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


class TestジョブIDの発行:
    def test_日時と乱数からなるIDが払い出される(self):
        job_id = jobs.new_job_id()

        assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", job_id), (
            f"想定外の書式: {job_id}"
        )

    def test_連続して発行してもIDが衝突しない(self):
        ids = {jobs.new_job_id() for _ in range(20)}

        assert len(ids) == 20


class TestジョブIDの検証:
    """パストラバーサル対策。外部から渡る job_id を境界で弾く。"""

    def test_正当なIDはクラスタ保存先の直下に解決される(self):
        path = jobs.job_dir("20260727-120000-abc123")

        assert path.parent == jobs.CLUSTERS_ROOT
        assert path.name == "20260727-120000-abc123"

    @pytest.mark.parametrize(
        "invalid_id",
        ["../etc", "a/b", "/absolute", "..", "job id", "job.id", ""],
    )
    def test_不正なIDはValueErrorになる(self, invalid_id):
        with pytest.raises(ValueError, match="無効な job_id"):
            jobs.job_dir(invalid_id)


class Testジョブの保存と読み込み:
    def test_保存したジョブがそのまま読み戻せる(self):
        job_id = "20260727-120000-abc123"
        job = {"csv_path": "/tmp/x.csv", "csv_filename": "x.csv", "clusters": []}

        jobs.save_job(job_id, job)

        assert jobs.load_job(job_id) == job

    def test_保存先にjob_jsonが作られる(self):
        job_id = "20260727-120000-abc123"

        jobs.save_job(job_id, {"csv_filename": "x.csv"})

        saved = json.loads(
            (jobs.CLUSTERS_ROOT / job_id / "job.json").read_text(encoding="utf-8")
        )
        assert saved == {"csv_filename": "x.csv"}

    def test_メモリキャッシュが空でもディスクから読み戻せる(self, monkeypatch):
        job_id = "20260727-120000-abc123"
        jobs.save_job(job_id, {"csv_filename": "x.csv"})
        monkeypatch.setattr(jobs, "_JOBS", {})

        assert jobs.load_job(job_id) == {"csv_filename": "x.csv"}

    def test_旧フォーマットのジョブはファイル名が補完される(self):
        job_id = "20260727-120000-abc123"
        job_dir = jobs.CLUSTERS_ROOT / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "job.json").write_text(
            json.dumps({"csv_path": "/tmp/古い.csv"}), encoding="utf-8"
        )

        job = jobs.load_job(job_id)

        assert job["csv_filename"] == "古い.csv"

    def test_存在しないジョブはNoneを返す(self):
        assert jobs.load_job("20260727-999999-zzzzzz") is None

    def test_不正なジョブIDは例外ではなくNoneを返す(self):
        # ルート側で 404 相当の扱いにするため、読み込みでは例外にしない
        assert jobs.load_job("../etc") is None


class Test話者名の再ラベル:
    def test_一致した話者名が新しい名前に置換される(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(
            csv_path,
            [
                ["start", "end", "speaker", "text", "cosine_distance"],
                ["00:00:01:000", "00:00:02:000", "Unknown_1", "こんにちは", "1.000000"],
                ["00:00:03:000", "00:00:04:000", "太郎", "やあ", "0.300000"],
            ],
        )

        replaced = jobs.relabel_csv(csv_path, {"Unknown_1": ("花子", 0.25)})

        rows = _read_csv(csv_path)
        assert replaced == 1
        assert rows[1][2] == "花子"
        assert rows[1][4] == "0.250000", "距離は小数6桁で書き戻される"
        assert rows[2][2] == "太郎", "対象外の行は変更されない"

    def test_距離がNoneのときは距離列が空になる(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(
            csv_path,
            [
                ["speaker", "cosine_distance"],
                ["Unknown_1", "1.000000"],
            ],
        )

        jobs.relabel_csv(csv_path, {"Unknown_1": ("花子", None)})

        assert _read_csv(csv_path)[1] == ["花子", ""]

    def test_複数行が一括で置換される(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(
            csv_path,
            [["speaker"], ["Unknown_1"], ["Unknown_1"], ["Unknown_2"]],
        )

        replaced = jobs.relabel_csv(csv_path, {"Unknown_1": ("花子", None)})

        assert replaced == 2

    def test_距離列が無いCSVでも話者名だけ置換される(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(csv_path, [["speaker", "text"], ["Unknown_1", "こんにちは"]])

        replaced = jobs.relabel_csv(csv_path, {"Unknown_1": ("花子", 0.25)})

        assert replaced == 1
        assert _read_csv(csv_path)[1] == ["花子", "こんにちは"]

    def test_話者列が無いCSVは何も変更しない(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(csv_path, [["start", "text"], ["00:00:01:000", "こんにちは"]])
        before = csv_path.read_bytes()

        replaced = jobs.relabel_csv(csv_path, {"Unknown_1": ("花子", None)})

        assert replaced == 0
        assert csv_path.read_bytes() == before

    def test_空のマッピングでは何もしない(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(csv_path, [["speaker"], ["Unknown_1"]])
        before = csv_path.read_bytes()

        assert jobs.relabel_csv(csv_path, {}) == 0
        assert csv_path.read_bytes() == before

    def test_CSVが存在しないときは0を返す(self, tmp_path):
        assert jobs.relabel_csv(tmp_path / "missing.csv", {"a": ("b", None)}) == 0

    def test_ヘッダだけのCSVは0行置換になる(self, tmp_path):
        csv_path = tmp_path / "result.csv"
        _write_csv(csv_path, [["speaker", "text"]])

        assert jobs.relabel_csv(csv_path, {"Unknown_1": ("花子", None)}) == 0
