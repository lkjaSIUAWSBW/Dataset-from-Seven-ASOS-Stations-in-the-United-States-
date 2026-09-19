"""比较 Raw dataset 和 Processed dataset 中共同观测记录的原始值。"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path(r"D:\桌面\有用的论文思路\scientific data\放在github上的\Raw dataset")
PROCESSED_DIR = Path(r"D:\桌面\有用的论文思路\scientific data\放在github上的\Processed dataset")
OUTPUT_FILE = Path(__file__).resolve().parent / "原始值保持验证结果.csv"
TOL = 1e-9
FIELDS = ["station", "station_name", "lat", "lon", "valid(UTC)", "tmpf", "sknt", "drct", "pres1", "pres2", "pres3"]
NUMERIC = {"lat", "lon", "tmpf", "sknt", "drct", "pres1", "pres2", "pres3"}
KEY = ("station", "valid(UTC)")


def clean(v):
    return "" if v is None else str(v).replace("\ufeff", "").strip()


def number(v):
    try:
        x = float(clean(v))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def equal(field, a, b):
    if field not in NUMERIC:
        return clean(a) == clean(b), None
    x, y = number(a), number(b)
    if x is None or y is None:
        return x is None and y is None, None
    d = abs(x - y)
    return d <= TOL, d


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"文件没有表头：{path}")
        for row in reader:
            if None in row:
                raise ValueError(f"文件存在列数异常：{path}")
            yield {clean(k): clean(v) for k, v in row.items()}


def validate(raw_path, processed_path):
    raw, raw_counts = {}, Counter()
    processed, processed_counts = {}, Counter()
    for row in rows(raw_path):
        key = tuple(row.get(k, "") for k in KEY)
        raw_counts[key] += 1
        raw.setdefault(key, row)
    for row in rows(processed_path):
        key = tuple(row.get(k, "") for k in KEY)
        processed_counts[key] += 1
        processed.setdefault(key, row)

    common = set(raw) & set(processed)
    raw_missing = set(raw) - set(processed)
    mismatches = Counter()
    maximum = defaultdict(float)
    examples = {}
    comparisons = 0
    for key in common:
        for field in FIELDS:
            if field not in raw[key] or field not in processed[key]:
                mismatches[field] += 1
                examples.setdefault(field, "处理后文件缺少该字段")
                continue
            comparisons += 1
            ok, diff = equal(field, raw[key][field], processed[key][field])
            if not ok:
                mismatches[field] += 1
                if diff is not None:
                    maximum[field] = max(maximum[field], diff)
                examples.setdefault(field, f"键={key}; 原始值={raw[key][field]}; 处理后值={processed[key][field]}")

    raw_count = sum(raw_counts.values())
    processed_count = sum(processed_counts.values())
    duplicate_raw = sum(max(0, n - 1) for n in raw_counts.values())
    duplicate_processed = sum(max(0, n - 1) for n in processed_counts.values())
    passed = sum(mismatches.values()) == 0
    common_info = {
        "记录类型": "字段汇总", "站点": raw_path.stem, "原始记录数": raw_count,
        "处理后记录数": processed_count, "原始唯一时间键数": len(raw),
        "处理后唯一时间键数": len(processed), "共同时间键数": len(common),
        "原始记录未在处理后找到数": sum(raw_counts[k] for k in raw_missing),
        "处理后新增时间键数": len(set(processed) - set(raw)),
        "原始重复键产生的重复记录数": duplicate_raw,
        "处理后重复键产生的重复记录数": duplicate_processed,
        "比较字段值总数": comparisons,
        "验证是否通过": "是" if passed else "否",
    }
    result = []
    for field in FIELDS:
        result.append({**common_info, "字段": field, "不一致值数量": mismatches[field],
                       "最大绝对误差": f"{maximum[field]:.15g}" if maximum[field] else "0",
                       "不一致示例": examples.get(field, "")})
    return result


def main():
    if not RAW_DIR.exists() or not PROCESSED_DIR.exists():
        raise FileNotFoundError("请检查 Raw dataset 和 Processed dataset 路径。")
    raw_files = {p.name: p for p in RAW_DIR.glob("*.csv")}
    processed_files = {p.name: p for p in PROCESSED_DIR.glob("*.csv")}
    if set(raw_files) != set(processed_files):
        raise FileNotFoundError("两套数据中的 CSV 文件名不完全一致。")
    output = []
    for name in sorted(raw_files):
        output.extend(validate(raw_files[name], processed_files[name]))
    headers = ["记录类型", "站点", "字段", "原始记录数", "处理后记录数", "原始唯一时间键数",
               "处理后唯一时间键数", "共同时间键数", "原始记录未在处理后找到数", "处理后新增时间键数",
               "原始重复键产生的重复记录数", "处理后重复键产生的重复记录数", "比较字段值总数",
               "不一致值数量", "最大绝对误差", "不一致示例", "验证是否通过"]
    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output)
    print(f"验证完成，结果已保存到：{OUTPUT_FILE}")
    print("判定规则：共同时间键中的原始字段全部一致即通过；未匹配边界记录仅单独统计。")


if __name__ == "__main__":
    main()
