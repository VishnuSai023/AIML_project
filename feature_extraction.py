from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"C:\AIML\5. Battery Data Set"
)

OUTPUT_DIR = DATASET_DIR / "NASA_Processed"

EOL_SOH = 70.0

SEARCH_SUBFOLDERS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# COMPACT, CONSISTENT OUTPUT COLUMN NAMES
# ============================================================
import re

NAME_REPLACEMENTS = [
    ("rectified_impedance", "rimp"),
    ("impedance_magnitude", "imp_mag"),
    ("impedance_phase", "imp_phase"),
    ("voltage_measured", "v"),
    ("current_measured", "i"),
    ("temperature_measured", "temp"),
    ("voltage_charge", "v_chg"),
    ("current_charge", "i_chg"),
    ("voltage_load", "v_load"),
    ("current_load", "i_load"),
    ("ambient_temperature", "amb_temp"),
    ("ambient_temp", "amb_temp"),
    ("discharge_cycle", "cyc"),
    ("charge_cycle", "chg_cyc"),
    ("record_index", "rec"),
    ("record_type", "type"),
    ("reference_capacity", "cap_ref"),
    ("capacity_loss_pct", "cap_loss_pct"),
    ("capacity_change", "cap_chg"),
    ("capacity_past_mean", "cap_pm"),
    ("capacity_past_std", "cap_ps"),
    ("capacity_past_slope", "cap_slope"),
    ("capacity_lag", "cap_lag"),
    ("capacity", "cap"),
    ("soh_past_mean", "soh_pm"),
    ("soh_change", "soh_chg"),
    ("rul_observed", "rul_obs"),
    ("eol_cycle_observed", "eol_cyc"),
    ("eol_observed", "eol_hit"),
    ("impedance_number", "imp_n"),
    ("voltage", "v"),
    ("current", "i"),
    ("temperature", "temp"),
    ("rectified", "rect"),
    ("magnitude", "mag"),
    ("previous", "prev"),
    ("reference", "ref"),
    ("observed", "obs"),
    ("percentage", "pct"),
    ("signed", "sgn"),
    ("absolute", "abs"),
    ("fraction", "frac"),
    ("correlation", "corr"),
    ("duration", "dur"),
    ("energy", "eng"),
    ("charge", "chg"),
    ("discharge", "dis"),
    ("samples", "n"),
    ("median", "med"),
    ("standard_deviation", "sd"),
    ("std", "sd"),
    ("slope", "slp"),
    ("change", "chg"),
]

# Exact output names requested for the cycle-level feature table.
# These are applied before generic shortening so the result is predictable.
EXACT_NAME_MAP = {
    "prev_rectified_impedance_magnitude_std": "imp_rect_std",
    "prev_rectified_impedance_magnitude_mean": "imp_rect_mean",
    "prev_impedance_magnitude_mean": "imp_mean",
    "prev_impedance_magnitude_std": "imp_std",
    "prev_impedance_phase_mean": "imp_phase",
    "prev_charge_voltage_load_mean": None,  # invalid for charge records; never export
    "voltage_at_50pct": "v_50",
    "temperature_peak_time": "temp_peak_pos",
    "average_discharge_power": "power_mean",
    "capacity_change_pct": "cap_change_pct",
    "capacity_fade_rate": "cap_fade_rate",
    "discharge_duration": "duration",
    "duration": "duration",
}

def compact_name(name):
    original = str(name).strip().lower()
    if original in EXACT_NAME_MAP:
        return EXACT_NAME_MAP[original]
    name = original
    for old, new in NAME_REPLACEMENTS:
        name = name.replace(old, new)
    name = re.sub(r"_+", "_", name).strip("_")
    return name

def compact_columns(df):
    if df.empty and len(df.columns) == 0:
        return df
    # Drop explicitly invalid fields before renaming, then remove any columns
    # that have no values anywhere in the combined output.
    keep = [c for c in df.columns if compact_name(c) is not None]
    out = df.loc[:, keep].copy()
    renamed = [compact_name(c) for c in out.columns]
    seen = {}
    unique = []
    for name in renamed:
        n = seen.get(name, 0)
        seen[name] = n + 1
        unique.append(name if n == 0 else f"{name}_{n+1}")
    out.columns = unique
    return out.dropna(axis=1, how="all")


# ============================================================
# MATLAB STRUCTURE HELPERS
# ============================================================

def get_field(obj, name, default=None):
    """Safely read a field from a MATLAB struct."""
    if obj is None:
        return default

    if hasattr(obj, name):
        return getattr(obj, name)

    if isinstance(obj, dict):
        return obj.get(name, default)

    return default


def scalar(value):
    """Convert a MATLAB scalar to a Python float."""
    if value is None:
        return np.nan

    try:
        arr = np.asarray(value).squeeze()

        if arr.size == 0:
            return np.nan

        result = arr.item()

        if np.iscomplexobj(result):
            return float(np.abs(result))

        result = float(result)
        return result if np.isfinite(result) else np.nan

    except (TypeError, ValueError):
        return np.nan


def numeric_array(value):
    """Convert a MATLAB signal to a flat numeric array."""
    if value is None:
        return np.array([], dtype=float)

    try:
        arr = np.asarray(value).squeeze()

        if arr.size == 0:
            return np.array([], dtype=float)

        arr = np.asarray(arr, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    except (TypeError, ValueError):
        return np.array([], dtype=float)


def get_cycles(mat):
    """
    Find a battery struct containing a 'cycle' field.
    Supports files such as B0005.mat.
    """
    for variable_name, value in mat.items():
        if variable_name.startswith("__"):
            continue

        cycles = get_field(value, "cycle")

        if cycles is not None:
            return variable_name, np.asarray(cycles).reshape(-1)

    return None, None


# ============================================================
# SIGNAL FEATURE ENGINEERING
# ============================================================

def signal_features(values, prefix):
    """Compact signal summaries; avoid dozens of redundant quantiles/extrema."""
    x = numeric_array(values)
    result = {
        f"{prefix}_samples": int(len(x)),
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_slope": np.nan,
        f"{prefix}_first": np.nan,
        f"{prefix}_last": np.nan,
        f"{prefix}_change": np.nan,
        f"{prefix}_rms": np.nan,
    }
    if not len(x):
        return result
    result.update({
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_first": float(x[0]),
        f"{prefix}_last": float(x[-1]),
        f"{prefix}_change": float(x[-1] - x[0]),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
    })
    if len(x) >= 2:
        result[f"{prefix}_slope"] = float(np.polyfit(np.linspace(0, 1, len(x)), x, 1)[0])
    return result


def extract_signal_data(data, record_type):
    """
    Extract features from a charge or discharge record.
    Integrals use the recorded Time signal, assumed to be seconds.
    """
    features = {}

    voltage = numeric_array(
        get_field(data, "Voltage_measured")
    )
    current = numeric_array(
        get_field(data, "Current_measured")
    )
    temperature = numeric_array(
        get_field(data, "Temperature_measured")
    )
    time_values = numeric_array(
        get_field(data, "Time")
    )

    features.update(signal_features(voltage, "voltage"))
    features.update(signal_features(current, "current"))
    features.update(signal_features(temperature, "temp"))

    # Physically interpretable features not captured by simple summaries.
    features["voltage_drop"] = float(voltage[0] - voltage[-1]) if len(voltage) >= 2 else np.nan
    features["temp_rise"] = float(temperature[-1] - temperature[0]) if len(temperature) >= 2 else np.nan
    features["temp_peak_fraction"] = (float(np.argmax(temperature) / (len(temperature) - 1))
                                      if len(temperature) >= 2 else np.nan)
    features["current_abs_mean"] = float(np.mean(np.abs(current))) if len(current) else np.nan
    features["current_rms"] = float(np.sqrt(np.mean(current ** 2))) if len(current) else np.nan
    features["voltage_at_10pct"] = float(np.interp(0.10, np.linspace(0, 1, len(voltage)), voltage)) if len(voltage) >= 2 else np.nan
    features["voltage_at_50pct"] = float(np.interp(0.50, np.linspace(0, 1, len(voltage)), voltage)) if len(voltage) >= 2 else np.nan
    features["voltage_at_90pct"] = float(np.interp(0.90, np.linspace(0, 1, len(voltage)), voltage)) if len(voltage) >= 2 else np.nan
    features["voltage_curve_area"] = float(np.trapezoid(voltage, np.linspace(0, 1, len(voltage)))) if len(voltage) >= 2 else np.nan
    features["temp_above_35_fraction"] = float(np.mean(temperature > 35.0)) if len(temperature) else np.nan
    features["current_sign_changes"] = int(np.sum(np.diff(np.sign(current)) != 0)) if len(current) >= 2 else 0

    if len(time_values) >= 2:
        duration = float(time_values[-1] - time_values[0])
        features["duration"] = duration if duration >= 0 else np.nan
    else:
        features["duration"] = np.nan

    # Correlation requires aligned signal samples.
    def add_correlation(a, b, name):
        n = min(len(a), len(b))

        if n >= 3:
            aa, bb = a[:n], b[:n]

            if np.std(aa) > 0 and np.std(bb) > 0:
                features[name] = float(np.corrcoef(aa, bb)[0, 1])
                return

        features[name] = np.nan

    add_correlation(
        voltage, temperature, "voltage_temp_corr"
    )
    add_correlation(
        voltage, current, "voltage_current_corr"
    )
    add_correlation(
        current, temperature, "current_temp_corr"
    )

    # Integrate only when signal and time lengths match.
    # Current convention can vary; retain signed and absolute results.
    if (
        len(time_values) >= 2
        and len(current) == len(time_values)
    ):
        dt = np.diff(time_values)

        if np.all(dt >= 0):
            charge_as = np.trapezoid(current, time_values)
            abs_charge_as = np.trapezoid(
                np.abs(current), time_values
            )

            features["charge_signed_ah"] = charge_as / 3600.0
            features["charge_abs_ah"] = abs_charge_as / 3600.0
        else:
            features["charge_signed_ah"] = np.nan
            features["charge_abs_ah"] = np.nan
    else:
        features["charge_signed_ah"] = np.nan
        features["charge_abs_ah"] = np.nan

    if (
        len(time_values) >= 2
        and len(voltage) == len(time_values)
        and len(current) == len(time_values)
    ):
        power = voltage * current

        if np.all(np.diff(time_values) >= 0):
            energy_signed_wh = np.trapezoid(
                power, time_values
            ) / 3600.0

            energy_abs_wh = np.trapezoid(
                np.abs(power), time_values
            ) / 3600.0

            features["energy_signed_wh"] = energy_signed_wh
            features["energy_abs_wh"] = energy_abs_wh
        else:
            features["energy_signed_wh"] = np.nan
            features["energy_abs_wh"] = np.nan
    else:
        features["energy_signed_wh"] = np.nan
        features["energy_abs_wh"] = np.nan

    # Record-specific fields
    features["capacity"] = np.nan
    features["current_charge_mean"] = np.nan
    features["current_load_mean"] = np.nan
    features["voltage_charge_mean"] = np.nan
    features["voltage_load_mean"] = np.nan

    if record_type == "discharge":
        features["capacity"] = scalar(
            get_field(data, "Capacity")
        )

        features["current_load_mean"] = safe_mean(
            get_field(data, "Current_load")
        )
        features["voltage_load_mean"] = safe_mean(
            get_field(data, "Voltage_load")
        )

    elif record_type == "charge":
        features["current_charge_mean"] = safe_mean(
            get_field(data, "Current_charge")
        )
        features["voltage_charge_mean"] = safe_mean(
            get_field(data, "Voltage_charge")
        )

    return features


def safe_mean(value):
    arr = numeric_array(value)
    return float(np.mean(arr)) if len(arr) else np.nan


# ============================================================
# IMPEDANCE FEATURES
# ============================================================

def impedance_features(data):
    """
    Compact impedance descriptors. NASA impedance arrays are frequency/
    measurement vectors; keep physically interpretable summaries rather than
    exporting a large family of near-duplicate statistics.
    """
    features = {
        "re": scalar(get_field(data, "Re")),
        "rct": scalar(get_field(data, "Rct")),
        "sense_current": scalar(get_field(data, "Sense_current")),
        "battery_current": scalar(get_field(data, "Battery_current")),
        "current_ratio": scalar(get_field(data, "Current_ratio")),
    }

    for field, prefix in [
        ("Battery_impedance", "impedance"),
        ("Rectified_Impedance", "rectified_impedance"),
    ]:
        raw = get_field(data, field)
        features[f"{prefix}_magnitude_mean"] = np.nan
        features[f"{prefix}_magnitude_std"] = np.nan
        features[f"{prefix}_phase_mean"] = np.nan

        if raw is None:
            continue

        try:
            z = np.asarray(raw).reshape(-1)
            z = z[np.isfinite(z)]
            if not len(z):
                continue

            features[f"{prefix}_magnitude_mean"] = float(np.mean(np.abs(z)))
            features[f"{prefix}_magnitude_std"] = float(np.std(np.abs(z)))
            features[f"{prefix}_phase_mean"] = float(np.mean(np.angle(z)))
        except (TypeError, ValueError):
            continue

    return features


# ============================================================
# PROCESS ONE BATTERY FILE
# ============================================================

def process_battery(mat_path):
    battery_id = mat_path.stem

    mat = loadmat(
        mat_path,
        struct_as_record=False,
        squeeze_me=True
    )

    variable_name, cycles = get_cycles(mat)

    if cycles is None:
        raise ValueError(
            "Could not find a MATLAB struct containing 'cycle'"
        )

    events = []
    impedance_rows = []

    discharge_number = 0
    charge_number = 0
    impedance_number = 0

    for record_index, cycle in enumerate(cycles, start=1):
        record_type = get_field(cycle, "type")
        record_type = str(record_type).strip().lower()

        data = get_field(cycle, "data")

        if data is None:
            continue

        base = {
            "battery_id": battery_id,
            "record_index": record_index,
            "record_type": record_type,
            "ambient_temp": scalar(
                get_field(cycle, "ambient_temperature")
            ),
        }

        if record_type == "discharge":
            discharge_number += 1

            features = extract_signal_data(data, "discharge")
            base["discharge_cycle"] = discharge_number
            base["charge_cycle"] = np.nan

            events.append({**base, **features})

        elif record_type == "charge":
            charge_number += 1

            features = extract_signal_data(data, "charge")
            base["discharge_cycle"] = np.nan
            base["charge_cycle"] = charge_number

            events.append({**base, **features})

        elif record_type == "impedance":
            impedance_number += 1

            row = {
                **base,
                "impedance_number": impedance_number,
                **impedance_features(data),
            }
            impedance_rows.append(row)

    event_df = pd.DataFrame(events)
    impedance_df = pd.DataFrame(impedance_rows)
    # Defensive guard: prevent duplicate labels from breaking as-of joins.
    event_df = event_df.loc[:, ~event_df.columns.duplicated()].copy()
    impedance_df = impedance_df.loc[:, ~impedance_df.columns.duplicated()].copy()

    if event_df.empty:
        raise ValueError("No charge/discharge records extracted")

    # --------------------------------------------------------
    # Create one row per discharge cycle
    # --------------------------------------------------------

    discharge_df = event_df[
        event_df["record_type"] == "discharge"
    ].copy()

    discharge_df = discharge_df.sort_values(
        "record_index"
    ).reset_index(drop=True)

    if discharge_df.empty:
        raise ValueError("No discharge records found")

    valid_capacity = discharge_df["capacity"].dropna()

    if valid_capacity.empty or valid_capacity.iloc[0] <= 0:
        reference_capacity = np.nan
    else:
        reference_capacity = float(valid_capacity.iloc[0])

    discharge_df["reference_capacity"] = reference_capacity

    discharge_df["soh"] = (
        discharge_df["capacity"] / reference_capacity * 100.0
        if np.isfinite(reference_capacity)
        else np.nan
    )

    discharge_df["capacity_loss_pct"] = (
        100.0 - discharge_df["soh"]
    )

    # Capacity changes between consecutive observed discharges.
    discharge_df["capacity_change"] = (
        discharge_df["capacity"].diff()
    )

    discharge_df["soh_change"] = discharge_df["soh"].diff()

    # Previous-cycle values only: valid historical predictors.
    discharge_df["capacity_lag1"] = (
        discharge_df["capacity"].shift(1)
    )
    discharge_df["capacity_lag2"] = (
        discharge_df["capacity"].shift(2)
    )
    discharge_df["soh_lag1"] = (
        discharge_df["soh"].shift(1)
    )
    discharge_df["soh_lag2"] = (
        discharge_df["soh"].shift(2)
    )

    # Rolling values use prior cycles only, excluding current target.
    past_capacity = discharge_df["capacity"].shift(1)

    discharge_df["capacity_past_mean5"] = (
        past_capacity.rolling(5, min_periods=2).mean()
    )
    discharge_df["capacity_past_std5"] = (
        past_capacity.rolling(5, min_periods=2).std()
    )

    discharge_df["soh_past_mean5"] = (
        discharge_df["soh"].shift(1).rolling(
            5, min_periods=2
        ).mean()
    )

    discharge_df["capacity_past_slope5"] = np.nan

    for i in range(len(discharge_df)):
        start = max(0, i - 5)
        history = discharge_df["capacity"].iloc[start:i].dropna()

        if len(history) >= 2:
            discharge_df.loc[
                i, "capacity_past_slope5"
            ] = np.polyfit(
                np.arange(len(history)),
                history.to_numpy(),
                1
            )[0]

    # --------------------------------------------------------
    # Observed EOL: only report if 70% was reached in the data.
    # --------------------------------------------------------

    below_eol = (
        discharge_df["soh"].notna()
        & (discharge_df["soh"] <= EOL_SOH)
    )

    if below_eol.any():
        first_eol_index = int(
            np.flatnonzero(below_eol.to_numpy())[0]
        )
        eol_cycle = int(
            discharge_df.loc[first_eol_index, "discharge_cycle"]
        )

        discharge_df["eol_observed"] = True
        discharge_df["eol_cycle_observed"] = eol_cycle

        # Observed RUL for historical analysis only.
        discharge_df["rul_observed"] = (
            eol_cycle - discharge_df["discharge_cycle"]
        ).clip(lower=0)
    else:
        discharge_df["eol_observed"] = False
        discharge_df["eol_cycle_observed"] = np.nan
        discharge_df["rul_observed"] = np.nan

    # --------------------------------------------------------
    # Attach the latest preceding charge record.
    # This is a backward-looking as-of join.
    # --------------------------------------------------------

    charge_df = event_df[
        event_df["record_type"] == "charge"
    ].copy()

    if not charge_df.empty:
        charge_df = charge_df.loc[:, ~charge_df.columns.duplicated()].copy()
        charge_df = charge_df.sort_values("record_index")

        # Only retain features that are meaningful for a CHARGE record.
        # Do not join discharge-only fields (for example voltage_load_mean)
        # from charge rows; those are structurally missing by definition.
        charge_feature_allowlist = [
            "voltage_mean",
            "voltage_std",
            "voltage_slope",
            "current_mean",
            "current_rms",
            "current_std",
            "temp_mean",
            "temp_rise",
            "temp_slope",
            "duration",
            "charge_signed_ah",
            "charge_abs_ah",
            "energy_signed_wh",
            "energy_abs_wh",
            "current_charge_mean",
            "voltage_charge_mean",
        ]
        charge_cols = [
            c for c in charge_feature_allowlist
            if c in charge_df.columns
        ]

        charge_df = charge_df[
            ["record_index"] + charge_cols
        ].copy()

        charge_df = charge_df.rename(
            columns={
                c: f"prev_charge_{c}"
                for c in charge_cols
                if c != "record_index"
            }
        )

        discharge_df = pd.merge_asof(
            discharge_df.sort_values("record_index"),
            charge_df.sort_values("record_index"),
            on="record_index",
            direction="backward",
            allow_exact_matches=False,
        )

    # --------------------------------------------------------
    # Attach latest preceding impedance measurement.
    # --------------------------------------------------------

    if not impedance_df.empty:
        impedance_df = impedance_df.loc[:, ~impedance_df.columns.duplicated()].copy()
        impedance_df = impedance_df.sort_values("record_index")

        # Compact, interpretable impedance set. The other raw impedance
        # fields are kept in NASA_impedance_features.csv, not widened into
        # dozens of columns in the cycle table.
        impedance_feature_allowlist = [
            "re",
            "rct",
            "impedance_magnitude_mean",
            "impedance_magnitude_std",
            "impedance_phase_mean",
            "rectified_impedance_magnitude_mean",
            "rectified_impedance_magnitude_std",
            "rectified_impedance_phase_mean",
        ]
        imp_cols = [
            c for c in impedance_feature_allowlist
            if c in impedance_df.columns
        ]

        imp_join = impedance_df[
            ["record_index"] + imp_cols
        ].copy()

        imp_join = imp_join.rename(
            columns={
                c: f"prev_{c}"
                for c in imp_cols
                if c != "record_index"
            }
        )

        discharge_df = pd.merge_asof(
            discharge_df.sort_values("record_index"),
            imp_join.sort_values("record_index"),
            on="record_index",
            direction="backward",
            allow_exact_matches=False,
        )

    return discharge_df, impedance_df, event_df


# ============================================================
# BATCH PROCESS ALL FILES
# ============================================================

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {DATASET_DIR}"
        )

    if SEARCH_SUBFOLDERS:
        mat_files = sorted(DATASET_DIR.rglob("*.mat"))
    else:
        mat_files = sorted(DATASET_DIR.glob("*.mat"))

    # Avoid processing previously generated outputs.
    mat_files = [
        p for p in mat_files
        if OUTPUT_DIR not in p.parents
    ]

    # Keep all source files, but report repeated battery IDs for review; do not silently merge copies.
    duplicate_ids = sorted({p.stem for p in mat_files if sum(q.stem == p.stem for q in mat_files) > 1})
    if duplicate_ids:
        print("WARNING: repeated battery filenames found in different folders:", ", ".join(duplicate_ids))
        print("Review the data-quality report before treating these as distinct batteries.\n")

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found under {DATASET_DIR}"
        )

    print(f"Found {len(mat_files)} MATLAB files.")
    print("Starting batch feature extraction...\n")

    all_cycles = []
    all_impedance = []
    all_events = []
    reports = []

    for i, mat_path in enumerate(mat_files, start=1):
        print(
            f"[{i}/{len(mat_files)}] Processing "
            f"{mat_path.name}..."
        )

        try:
            cycle_df, imp_df, event_df = process_battery(mat_path)

            all_cycles.append(cycle_df)
            all_impedance.append(imp_df)
            all_events.append(event_df)

            valid_soh = cycle_df["soh"].dropna()
            valid_capacity = cycle_df["capacity"].dropna()

            eol_observed = bool(
                cycle_df["eol_observed"].iloc[0]
            )

            reports.append({
                "battery_id": mat_path.stem,
                "file": str(mat_path),
                "status": "success",
                "records": len(event_df),
                "discharge_cycles": len(cycle_df),
                "charge_records": int(
                    (event_df["record_type"] == "charge").sum()
                ),
                "impedance_records": len(imp_df),
                "capacity_missing": int(
                    cycle_df["capacity"].isna().sum()
                ),
                "capacity_min": (
                    valid_capacity.min()
                    if len(valid_capacity) else np.nan
                ),
                "capacity_max": (
                    valid_capacity.max()
                    if len(valid_capacity) else np.nan
                ),
                "soh_min": (
                    valid_soh.min()
                    if len(valid_soh) else np.nan
                ),
                "soh_max": (
                    valid_soh.max()
                    if len(valid_soh) else np.nan
                ),
                "eol_threshold_pct": EOL_SOH,
                "eol_observed": eol_observed,
                "error": "",
            })

        except Exception as exc:
            warnings.warn(
                f"Failed to process {mat_path.name}: {exc}"
            )

            reports.append({
                "battery_id": mat_path.stem,
                "file": str(mat_path),
                "status": "failed",
                "records": 0,
                "discharge_cycles": 0,
                "charge_records": 0,
                "impedance_records": 0,
                "capacity_missing": np.nan,
                "capacity_min": np.nan,
                "capacity_max": np.nan,
                "soh_min": np.nan,
                "soh_max": np.nan,
                "eol_threshold_pct": EOL_SOH,
                "eol_observed": False,
                "error": str(exc),
            })

    def combine(frames):
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    cycles = combine(all_cycles)
    impedance = combine(all_impedance)
    events = combine(all_events)
    report_df = pd.DataFrame(reports)

    # Compact every output table before writing. Calculations above retain
    # descriptive internal names; only exported CSV columns are renamed.
    cycles_out = compact_columns(cycles)
    impedance_out = compact_columns(impedance)
    events_out = compact_columns(events)
    report_out = compact_columns(report_df)

    cycles_out.to_csv(OUTPUT_DIR / "NASA_cycle_features.csv", index=False)
    impedance_out.to_csv(OUTPUT_DIR / "NASA_impedance_features.csv", index=False)
    events_out.to_csv(OUTPUT_DIR / "NASA_all_events.csv", index=False)
    report_out.to_csv(OUTPUT_DIR / "NASA_data_quality_summary.csv", index=False)

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    descriptions = {
        "battery_id": "Battery identifier derived from filename",
        "record_index": "Original position of record in the MATLAB cycle array",
        "discharge_cycle": "Consecutive discharge count within a battery",
        "capacity": "Capacity recorded for a discharge cycle, Ah",
        "reference_capacity": "First valid discharge capacity for this battery, Ah",
        "soh": "Capacity divided by reference capacity, multiplied by 100",
        "capacity_loss_pct": "100 minus SOH",
        "capacity_lag1": "Previous discharge capacity, Ah",
        "capacity_lag2": "Capacity from two discharge cycles earlier, Ah",
        "capacity_past_mean5": "Mean of up to five previous discharge capacities",
        "capacity_past_std5": "Standard deviation of previous discharge capacities",
        "capacity_past_slope5": "Linear trend in previous discharge capacities",
        "rul_observed": "Cycles to first observed 70% SOH crossing; not future prediction",
        "eol_observed": "Whether this battery reached 70% SOH in observed data",
        "duration": "Elapsed time in the record's Time signal, seconds",
        "charge_abs_ah": "Integral of absolute measured current over time, Ah",
        "charge_signed_ah": "Signed current integral over time, Ah",
        "energy_abs_wh": "Integral of absolute voltage-current product, Wh",
        "energy_signed_wh": "Signed voltage-current integral, Wh",
        "re": "Recorded impedance electrolyte resistance field",
        "rct": "Recorded charge-transfer resistance field",
    }

    dictionary = pd.DataFrame({
        "feature": cycles.columns,
        "description": [
            descriptions.get(
                c,
                "Engineered signal statistic or source measurement; inspect source units"
            )
            for c in cycles.columns
        ],
    })

    dictionary["feature"] = dictionary["feature"].map(compact_name)
    dictionary = dictionary.drop_duplicates(subset=["feature"], keep="first")
    dictionary.to_csv(OUTPUT_DIR / "NASA_feature_dictionary.csv", index=False)

    print("\n" + "=" * 60)
    print("BATCH EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Files discovered: {len(mat_files)}")
    print(
        f"Files processed successfully: "
        f"{(report_df['status'] == 'success').sum()}"
    )
    print(
        f"Files failed: {(report_df['status'] == 'failed').sum()}"
    )
    print(f"Combined discharge cycles: {len(cycles)}")
    print(f"Combined impedance records: {len(impedance)}")
    print(f"Output folder: {OUTPUT_DIR}")

    if not report_df.empty:
        print("\nProcessing summary:")
        print(
            report_df[
                [
                    "battery_id",
                    "status",
                    "discharge_cycles",
                    "impedance_records",
                    "eol_observed",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()