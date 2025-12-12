from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
import unittest


class Tlg8TrainingCliTest(unittest.TestCase):
    """TLG8 の学習系オプションが組み合わせ通りに動くかを検証する。"""

    def setUp(self) -> None:
        """テスト用の実行ファイルと入力画像を用意する。"""

        self.tlgconv = Path("build") / "tlgconv"
        if not self.tlgconv.exists():
            self.skipTest("build/tlgconv が見つかりません")
        self.input_image = Path("test/small_images/flat.bmp")
        if not self.input_image.exists():
            self.fail("入力画像が見つかりません")

    def run_command(self, output_dir: Path, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
        """tlgconv を実行して結果を返す。"""

        out_path = output_dir / "out.tlg"
        cmd = [str(self.tlgconv), str(self.input_image), str(out_path)] + extra_args
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_dump_training_only(self) -> None:
        """--tlg8-dump-training のみで JSON が書き出される。"""

        with tempfile.TemporaryDirectory(prefix="tlg8_dump_only_") as tmpdir:
            work = Path(tmpdir)
            dump_path = work / "dump.json"
            result = self.run_command(work, [f"--tlg8-dump-training={dump_path}"])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(dump_path.exists(), "学習ダンプが生成されていません")
            self.assertGreater(dump_path.stat().st_size, 0)

    def test_label_cache_only(self) -> None:
        """ラベルキャッシュ単独指定でも書き出せる。"""

        with tempfile.TemporaryDirectory(prefix="tlg8_label_only_") as tmpdir:
            work = Path(tmpdir)
            bin_path = work / "labels.bin"
            meta_path = work / "labels.json"
            result = self.run_command(
                work,
                [f"--label-cache-bin={bin_path}", f"--label-cache-meta={meta_path}"],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(bin_path.exists(), "ラベルキャッシュ bin が生成されていません")
            self.assertGreater(bin_path.stat().st_size, 0)
            self.assertTrue(meta_path.exists(), "ラベルキャッシュ meta が生成されていません")
            with meta_path.open("r", encoding="utf-8") as fp:
                meta = json.load(fp)
            self.assertGreater(meta.get("record_count", 0), 0)

    def test_dump_and_label_cache(self) -> None:
        """学習ダンプとラベルキャッシュの併用が機能する。"""

        with tempfile.TemporaryDirectory(prefix="tlg8_dump_and_label_") as tmpdir:
            work = Path(tmpdir)
            dump_path = work / "dump.json"
            bin_path = work / "labels.bin"
            meta_path = work / "labels.json"
            result = self.run_command(
                work,
                [
                    f"--tlg8-dump-training={dump_path}",
                    f"--label-cache-bin={bin_path}",
                    f"--label-cache-meta={meta_path}",
                ],
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(dump_path.exists())
            self.assertTrue(bin_path.exists())
            self.assertTrue(meta_path.exists())

    def test_invalid_label_cache_combo(self) -> None:
        """bin のみ指定など無効な組み合わせはエラーになる。"""

        with tempfile.TemporaryDirectory(prefix="tlg8_label_invalid_") as tmpdir:
            work = Path(tmpdir)
            bin_path = work / "labels.bin"
            result = self.run_command(work, [f"--label-cache-bin={bin_path}"])
            self.assertEqual(result.returncode, 2)
            self.assertIn("--label-cache-bin と --label-cache-meta は同時に指定してください", result.stderr)


if __name__ == "__main__":
    unittest.main()
