"""Check duplicate records and primary-variable quality in processed ASOS CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "Processed dataset"
RESULT_PATH = SCRIPT_DIR / "data_quality_check_results.csv"

STATION_FIELD = "station"
TIME_FIELD = "valid(UTC)"
PRIMARY_VARIABLES = ["tmpf", "sknt", "drct", "pres1", "pres2", "pres3"]
MISSING_TEXT = {"", "M", "NA", "N/A", "NULL", "NAN"}


def is_missing(series: pd.Series) -> pd.Series:
    """Recognize blank cells and common source missing-value codes."""
    return series.isna() | series.astype("string").str.strip().str.upper().isin(MISSING_TEXT)


def summarize_dataset(path: Path) -> dict[str, int | str]:
    data = pd.read_csv(path, dtype="string", keep_default_na=False, low_memory=False)
    required = {STATION_FIELD, TIME_FIELD, *PRIMARY_VARIABLES}
    absent = required.difference(data.columns)
    if absent:
        raise ValueError(f"{path.name} lacks required fields: {sorted(absent)}")

    
    
    if "MissingTimestamp" in data.columns:
        inserted_time_rows = data["MissingTimestamp"].str.strip().eq("1")
    else:
        inserted_time_rows = pd.Series(False, index=data.index)
    observations = data.loc[~inserted_time_rows].copy()

    duplicate_key = [STATION_FIELD, TIME_FIELD, *PRIMARY_VARIABLES]
    
    duplicate_rows = observations.duplicated(subset=duplicate_key, keep=False)
    duplicate_groups = int(observations.loc[duplicate_rows, duplicate_key].drop_duplicates().shape[0])

    wind_speed = pd.to_numeric(observations["sknt"].replace("M", pd.NA), errors="coerce")
    wind_direction = pd.to_numeric(observations["drct"].replace("M", pd.NA), errors="coerce")
    missing_primary = pd.concat([is_missing(observations[column]) for column in PRIMARY_VARIABLES], axis=1).any(axis=1)

    return {
        "dataset": path.name,
        "processed_rows": int(len(data)),
        "inserted_missing_timestamp_rows_excluded": int(inserted_time_rows.sum()),
        "original_observation_records_checked": int(len(observations)),
        "duplicate_record_groups": duplicate_groups,
        "duplicate_records_flagged": int(duplicate_rows.sum()),
        "WindSpeedNegative": int((wind_speed < 0).fillna(False).sum()),
        "WindDirectionInvalid": int(((wind_direction < 0) | (wind_direction > 360)).fillna(False).sum()),
        "MissingValueFlag": int(missing_primary.sum()),
    }


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {PROCESSED_DIR}")
    results = [summarize_dataset(path) for path in files]
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding="utf-8")
    print(f"Saved checks for {len(results)} datasets to {RESULT_PATH}")


if __name__ == "__main__":
    main()
