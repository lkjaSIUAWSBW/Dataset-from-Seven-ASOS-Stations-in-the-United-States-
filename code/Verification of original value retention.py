"""\u6bd4\u8f83 Raw dataset \u548c Processed dataset \u4e2d\u5171\u540c\u89c2\u6d4b\u8bb0\u5f55\u7684\u539f\u59cb\u503c。"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path(r"D:\\u684c\u9762\\u6709\u7528\u7684\u8bba\u6587\u601d\u8def\scientific data\\u653e\u5728github\u4e0a\u7684\Raw dataset")
PROCESSED_DIR = Path(r"D:\\u684c\u9762\\u6709\u7528\u7684\u8bba\u6587\u601d\u8def\scientific data\\u653e\u5728github\u4e0a\u7684\Processed dataset")
OUTPUT_FILE = Path(__file__).resolve().parent / "\u539f\u59cb\u503c\u4fdd\u6301\u9a8c\u8bc1\u7ed3\u679c.csv"
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
            raise ValueError(f"\u6587\u4ef6\u6ca1\u6709\u8868\u5934：{path}")
        for row in reader:
            if None in row:
                raise ValueError(f"\u6587\u4ef6\u5b58\u5728\u5217\u6570\u5f02\u5e38：{path}")
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
                examples.setdefault(field, "\u5904\u7406\u540e\u6587\u4ef6\u7f3a\u5c11\u8be5\u5b57\u6bb5")
                continue
            comparisons += 1
            ok, diff = equal(field, raw[key][field], processed[key][field])
            if not ok:
                mismatches[field] += 1
                if diff is not None:
                    maximum[field] = max(maximum[field], diff)
                examples.setdefault(field, f"\u952e={key}; \u539f\u59cb\u503c={raw[key][field]}; \u5904\u7406\u540e\u503c={processed[key][field]}")

    raw_count = sum(raw_counts.values())
    processed_count = sum(processed_counts.values())
    duplicate_raw = sum(max(0, n - 1) for n in raw_counts.values())
    duplicate_processed = sum(max(0, n - 1) for n in processed_counts.values())
    passed = sum(mismatches.values()) == 0
    common_info = {
        "\u8bb0\u5f55\u7c7b\u578b": "\u5b57\u6bb5\u6c47\u603b", "\u7ad9\u70b9": raw_path.stem, "\u539f\u59cb\u8bb0\u5f55\u6570": raw_count,
        "\u5904\u7406\u540e\u8bb0\u5f55\u6570": processed_count, "\u539f\u59cb\u552f\u4e00\u65f6\u95f4\u952e\u6570": len(raw),
        "\u5904\u7406\u540e\u552f\u4e00\u65f6\u95f4\u952e\u6570": len(processed), "\u5171\u540c\u65f6\u95f4\u952e\u6570": len(common),
        "\u539f\u59cb\u8bb0\u5f55\u672a\u5728\u5904\u7406\u540e\u627e\u5230\u6570": sum(raw_counts[k] for k in raw_missing),
        "\u5904\u7406\u540e\u65b0\u589e\u65f6\u95f4\u952e\u6570": len(set(processed) - set(raw)),
        "\u539f\u59cb\u91cd\u590d\u952e\u4ea7\u751f\u7684\u91cd\u590d\u8bb0\u5f55\u6570": duplicate_raw,
        "\u5904\u7406\u540e\u91cd\u590d\u952e\u4ea7\u751f\u7684\u91cd\u590d\u8bb0\u5f55\u6570": duplicate_processed,
        "\u6bd4\u8f83\u5b57\u6bb5\u503c\u603b\u6570": comparisons,
        "\u9a8c\u8bc1\u662f\u5426\u901a\u8fc7": "\u662f" if passed else "\u5426",
    }
    result = []
    for field in FIELDS:
        result.append({**common_info, "\u5b57\u6bb5": field, "\u4e0d\u4e00\u81f4\u503c\u6570\u91cf": mismatches[field],
                       "\u6700\u5927\u7edd\u5bf9\u8bef\u5dee": f"{maximum[field]:.15g}" if maximum[field] else "0",
                       "\u4e0d\u4e00\u81f4\u793a\u4f8b": examples.get(field, "")})
    return result


def main():
    if not RAW_DIR.exists() or not PROCESSED_DIR.exists():
        raise FileNotFoundError("\u8bf7\u68c0\u67e5 Raw dataset \u548c Processed dataset \u8def\u5f84。")
    raw_files = {p.name: p for p in RAW_DIR.glob("*.csv")}
    processed_files = {p.name: p for p in PROCESSED_DIR.glob("*.csv")}
    if set(raw_files) != set(processed_files):
        raise FileNotFoundError("\u4e24\u5957\u6570\u636e\u4e2d\u7684 CSV \u6587\u4ef6\u540d\u4e0d\u5b8c\u5168\u4e00\u81f4。")
    output = []
    for name in sorted(raw_files):
        output.extend(validate(raw_files[name], processed_files[name]))
    headers = ["\u8bb0\u5f55\u7c7b\u578b", "\u7ad9\u70b9", "\u5b57\u6bb5", "\u539f\u59cb\u8bb0\u5f55\u6570", "\u5904\u7406\u540e\u8bb0\u5f55\u6570", "\u539f\u59cb\u552f\u4e00\u65f6\u95f4\u952e\u6570",
               "\u5904\u7406\u540e\u552f\u4e00\u65f6\u95f4\u952e\u6570", "\u5171\u540c\u65f6\u95f4\u952e\u6570", "\u539f\u59cb\u8bb0\u5f55\u672a\u5728\u5904\u7406\u540e\u627e\u5230\u6570", "\u5904\u7406\u540e\u65b0\u589e\u65f6\u95f4\u952e\u6570",
               "\u539f\u59cb\u91cd\u590d\u952e\u4ea7\u751f\u7684\u91cd\u590d\u8bb0\u5f55\u6570", "\u5904\u7406\u540e\u91cd\u590d\u952e\u4ea7\u751f\u7684\u91cd\u590d\u8bb0\u5f55\u6570", "\u6bd4\u8f83\u5b57\u6bb5\u503c\u603b\u6570",
               "\u4e0d\u4e00\u81f4\u503c\u6570\u91cf", "\u6700\u5927\u7edd\u5bf9\u8bef\u5dee", "\u4e0d\u4e00\u81f4\u793a\u4f8b", "\u9a8c\u8bc1\u662f\u5426\u901a\u8fc7"]
    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output)
    print(f"\u9a8c\u8bc1\u5b8c\u6210，\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230：{OUTPUT_FILE}")
    print("\u5224\u5b9a\u89c4\u5219：\u5171\u540c\u65f6\u95f4\u952e\u4e2d\u7684\u539f\u59cb\u5b57\u6bb5\u5168\u90e8\u4e00\u81f4\u5373\u901a\u8fc7；\u672a\u5339\u914d\u8fb9\u754c\u8bb0\u5f55\u4ec5\u5355\u72ec\u7edf\u8ba1。")


if __name__ == "__main__":
    main()
