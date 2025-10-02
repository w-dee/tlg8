#!/usr/bin/env python3
from collections import Counter
from pathlib import Path
import re


# 分析結果からpredictorとfilterの出現回数を集計してヒストグラムを書き出す

def main() -> None:
    src_path = Path("analysis/vs.txt")
    dst_path = Path("analysis/vs_histogram.txt")
    pattern = re.compile(r"\b(predictor|filter)=(\d+)")
    predictor_counter: Counter[int] = Counter()
    filter_counter: Counter[int] = Counter()

    try:
        with src_path.open(encoding="utf-8") as src_file:
            for line in src_file:
                if not line.startswith("#"):
                    continue
                for key, value in pattern.findall(line):
                    number = int(value)
                    if key == "predictor":
                        predictor_counter[number] += 1
                    elif key == "filter":
                        filter_counter[number] += 1
    except FileNotFoundError:
        raise SystemExit("analysis/vs.txt が見つかりません")

    output_lines: list[str] = []
    output_lines.append("[predictor]\n")
    for number, count in sorted(predictor_counter.items()):
        output_lines.append(f"{number} {count}\n")

    output_lines.append("\n[filter]\n")
    for number, count in sorted(filter_counter.items()):
        output_lines.append(f"{number} {count}\n")

    dst_path.write_text("".join(output_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
