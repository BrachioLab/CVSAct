"""Reproducible surgeon-validation sampling for CVS-Act.

The sampling frame is the SAGES test split videos with at least one selected
ascending CVS transition. The existing audit_v11 videos are excluded before
drawing the new surgeon-validation sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .extraction import find_selected_ascending_clips, load_video_data
from .schemas import CRITERIA


SEED = 20260617
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "BrachioLab/cvs-act"
DATASET_CONFIG = "sages_trained_annotator"
DATASET_REVISION = "v1.0.0"
RAW_TEST_LABELS_DIR = REPO_ROOT / "data/raw/CVS_Challenge_SAGES_v1/test/labels"
TEST_METADATA_CSV = RAW_TEST_LABELS_DIR / "metadata.csv"
AUDIT_V11_DIR = (
    REPO_ROOT
    / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11"
)
SURGEON_ANNOTATION_ROOT = (
    REPO_ROOT / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_surgeon_annotations"
)
SURGEON_VALIDATION_ROOT = SURGEON_ANNOTATION_ROOT
COUNTRY_COLLAPSE_KEEP = {"0", "6"}
DEFAULT_ICG_FLOOR = 3
DEFAULT_ROBOTIC_FLOOR = 3
EXPECTED_ELIGIBLE_N = 168
EXPECTED_POOL_N = 138


@dataclass(frozen=True)
class WeightedSample:
    """Sample with post-stratification weights and design effect."""

    sample: pd.DataFrame
    weights: pd.Series
    design_effect: float


def _load_hf_video_names() -> set[str]:
    """Best-effort HF load for parity with public CVS-Act paths.

    The public CVS-Act HF dataset contains action-label rows, not the raw
    300-video challenge split. This helper is intentionally non-fatal; the
    local frame labels remain the source of truth for eligibility.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return set()
    try:
        ds = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION)
    except Exception:
        try:
            ds = load_dataset(str(REPO_ROOT / "hf_repos/cvs-act"), DATASET_CONFIG)
        except Exception:
            return set()
    names: set[str] = set()
    for split in ds:
        for row in ds[split]:
            vid = row.get("video_name") or row.get("video_id")
            if vid:
                names.add(str(vid))
    return names


def _read_metadata(metadata_csv: Path = TEST_METADATA_CSV) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")
    df = pd.read_csv(metadata_csv, dtype=str)
    required = {"video_name", "ioc", "icg", "robotic", "country", "device"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    return df[list(required)].copy()


def _transition_bits_from_video_data(video_data: Mapping[str, dict]) -> pd.DataFrame:
    selected = find_selected_ascending_clips(video_data)
    rows: list[dict] = []
    for video_name in sorted(video_data):
        row = {"video_name": video_name}
        for crit in CRITERIA:
            row[f"{crit}_bit"] = int(len(selected.get(video_name, {}).get(crit, [])) > 0)
        row["stratum"] = _format_stratum(row["c1_bit"], row["c2_bit"], row["c3_bit"])
        row["eligible"] = int(any(row[f"{crit}_bit"] for crit in CRITERIA))
        rows.append(row)
    return pd.DataFrame(rows)


def _format_stratum(c1_bit: int, c2_bit: int, c3_bit: int) -> str:
    return f"({int(c1_bit)},{int(c2_bit)},{int(c3_bit)})"


def load_eligible_pool(
    labels_dir: Path = RAW_TEST_LABELS_DIR,
    metadata_csv: Path = TEST_METADATA_CSV,
    verify_hf: bool = True,
) -> pd.DataFrame:
    """Load test labels, join metadata, compute eligibility, and return 168 videos.

    Rows are keyed by ``video_name``. No file order or metadata order is used
    as sample order. The HF dataset is loaded on a best-effort basis only to
    match repo conventions; raw eligibility is computed from local frame labels.
    """

    video_data = load_video_data(str(labels_dir))
    bits = _transition_bits_from_video_data(video_data)
    meta = _read_metadata(metadata_csv)
    df = bits.merge(meta, on="video_name", how="left", validate="one_to_one")
    if df[["ioc", "icg", "robotic", "country", "device"]].isna().any().any():
        missing = df.loc[df["country"].isna(), "video_name"].tolist()
        raise ValueError(f"Missing metadata for videos: {missing[:10]}")
    eligible = df.loc[df["eligible"] == 1].copy()
    eligible = add_collapsed_country(eligible)
    if verify_hf:
        eligible.attrs["hf_video_names_seen"] = len(_load_hf_video_names())
    if len(eligible) != EXPECTED_ELIGIBLE_N:
        raise AssertionError(f"Expected {EXPECTED_ELIGIBLE_N} eligible videos, got {len(eligible)}")
    if (eligible["stratum"] == "(0,0,0)").any():
        raise AssertionError("(0,0,0) stratum is non-empty in eligible pool")
    return eligible.sort_values("video_name").reset_index(drop=True)


def load_audit_v11_ids(audit_dir: Path = AUDIT_V11_DIR) -> list[str]:
    """Return audit_v11 video IDs from existing annotation JSON filenames."""

    if not audit_dir.exists():
        raise FileNotFoundError(f"Missing audit_v11 directory: {audit_dir}")
    return sorted(path.stem for path in audit_dir.glob("*.json"))


def exclude_audit_v11(df: pd.DataFrame, audit_ids: Iterable[str]) -> pd.DataFrame:
    """Exclude contaminated audit_v11 videos and assert the 138-video pool."""

    audit_set = set(map(str, audit_ids))
    overlap = set(df["video_name"]) & audit_set
    out = df.loc[~df["video_name"].isin(audit_set)].copy()
    if set(out["video_name"]) & audit_set:
        raise AssertionError("audit_v11 overlap remains after exclusion")
    if len(df) == EXPECTED_ELIGIBLE_N and len(overlap) != 30:
        raise AssertionError(f"Expected 30 audit_v11 overlaps, got {len(overlap)}")
    if len(out) != EXPECTED_POOL_N:
        raise AssertionError(f"Expected {EXPECTED_POOL_N} videos after audit_v11 exclusion, got {len(out)}")
    return out.sort_values("video_name").reset_index(drop=True)


def compute_transition_bits(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``c1_bit``, ``c2_bit``, ``c3_bit``, and ``stratum`` if needed."""

    out = df.copy()
    for crit in CRITERIA:
        col = f"{crit}_bit"
        if col not in out:
            raise ValueError(f"Missing transition bit column: {col}")
        out[col] = out[col].astype(int)
    out["stratum"] = [
        _format_stratum(c1, c2, c3)
        for c1, c2, c3 in zip(out["c1_bit"], out["c2_bit"], out["c3_bit"])
    ]
    return out


def add_collapsed_country(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve full country code and add ``country_collapsed`` in {0, 6, other}."""

    out = df.copy()
    out["country"] = out["country"].astype(str)
    out["country_collapsed"] = out["country"].where(
        out["country"].isin(COUNTRY_COLLAPSE_KEEP),
        "other",
    )
    return out


def _largest_remainder_quota(counts: pd.Series, n: int, min_per_nonzero: int = 0) -> dict[str, int]:
    counts = counts[counts > 0].sort_index()
    if counts.empty:
        return {}
    if min_per_nonzero and len(counts) * min_per_nonzero > n:
        raise ValueError("Too many populated cells for requested minimum quota")
    quotas = {str(k): min_per_nonzero for k in counts.index}
    remaining = n - sum(quotas.values())
    if remaining <= 0:
        return quotas
    weights = counts / counts.sum() * remaining
    floors = np.floor(weights).astype(int)
    for k, v in floors.items():
        quotas[str(k)] += int(v)
    leftover = n - sum(quotas.values())
    remainders = (weights - floors).sort_values(ascending=False)
    for k in remainders.index[:leftover]:
        quotas[str(k)] += 1
    return quotas


def _largest_remainder_quota_with_lower(
    counts: pd.Series,
    n: int,
    lower: Mapping[str, int],
) -> dict[str, int]:
    counts = counts[counts > 0].sort_index()
    quotas = {str(k): int(lower.get(str(k), 0)) for k in counts.index}
    if any(quotas[str(k)] <= 0 for k in counts.index):
        raise ValueError("Lower bounds must be positive for populated cells")
    if sum(quotas.values()) > n:
        raise ValueError("Lower bounds exceed requested sample size")
    remaining = n - sum(quotas.values())
    if remaining == 0:
        return quotas
    weights = counts / counts.sum() * remaining
    floors = np.floor(weights).astype(int)
    for k, v in floors.items():
        quotas[str(k)] += int(v)
    leftover = n - sum(quotas.values())
    remainders = (weights - floors).sort_values(ascending=False)
    for k in remainders.index[:leftover]:
        quotas[str(k)] += 1
    return quotas


def _country_quotas(df: pd.DataFrame, n: int) -> dict[str, int]:
    return _largest_remainder_quota(df["country_collapsed"].value_counts(), n, min_per_nonzero=1)


def _joint_quotas_with_margins(
    df: pd.DataFrame,
    stratum_quotas: Mapping[str, int],
    country_quotas: Mapping[str, int],
) -> dict[tuple[str, str], int]:
    """Allocate joint stratum x country quotas with exact marginal quotas."""

    counts = df.groupby(["stratum", "country_collapsed"]).size().to_dict()
    cells = sorted((str(s), str(c), int(n)) for (s, c), n in counts.items())
    if len(cells) > sum(stratum_quotas.values()):
        raise ValueError("More populated joint cells than requested sample size")

    quotas = {(s, c): 1 for s, c, _ in cells}
    row_remaining = {str(k): int(v) for k, v in stratum_quotas.items()}
    col_remaining = {str(k): int(v) for k, v in country_quotas.items()}
    capacities = {}
    for s, c, n in cells:
        row_remaining[s] -= 1
        col_remaining[c] -= 1
        capacities[(s, c)] = n - 1
    if any(v < 0 for v in row_remaining.values()) or any(v < 0 for v in col_remaining.values()):
        raise ValueError("Marginal quotas cannot cover all populated joint cells")

    rows = sorted({s for s, _, _ in cells})
    row_cells = {s: [(ss, c) for ss, c, _ in cells if ss == s] for s in rows}

    def distribute(total: int, caps: list[int], col_caps: list[int]) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []

        def rec(i: int, remaining: int, cur: list[int]) -> None:
            if i == len(caps):
                if remaining == 0:
                    out.append(tuple(cur))
                return
            max_v = min(caps[i], col_caps[i], remaining)
            for value in range(max_v + 1):
                cur.append(value)
                rec(i + 1, remaining - value, cur)
                cur.pop()

        rec(0, total, [])
        return out

    def backtrack(row_idx: int) -> bool:
        if row_idx == len(rows):
            return all(v == 0 for v in col_remaining.values())
        row = rows[row_idx]
        keys = row_cells[row]
        caps = [capacities[k] for k in keys]
        col_caps = [col_remaining[k[1]] for k in keys]
        for extra in distribute(row_remaining[row], caps, col_caps):
            for key, value in zip(keys, extra):
                quotas[key] += value
                col_remaining[key[1]] -= value
            if all(v >= 0 for v in col_remaining.values()) and backtrack(row_idx + 1):
                return True
            for key, value in zip(keys, extra):
                quotas[key] -= value
                col_remaining[key[1]] += value
        return False

    if not backtrack(0):
        raise ValueError("No feasible joint quota allocation for stratum and country margins")
    return quotas


def _check_constraints(
    sample: pd.DataFrame,
    country_quotas: Mapping[str, int] | None,
    icg_floor: int,
    robotic_floor: int,
) -> bool:
    if country_quotas is not None:
        got = sample["country_collapsed"].value_counts().to_dict()
        for key, want in country_quotas.items():
            if int(got.get(key, 0)) != int(want):
                return False
    if int((sample["icg"].astype(str) == "1").sum()) < icg_floor:
        return False
    if int((sample["robotic"].astype(str) == "1").sum()) < robotic_floor:
        return False
    return True


def stratified_sample(
    df: pd.DataFrame,
    n: int = 30,
    seed: int = SEED,
    icg_floor: int = DEFAULT_ICG_FLOOR,
    robotic_floor: int = DEFAULT_ROBOTIC_FLOOR,
    country_balance: bool = True,
) -> pd.DataFrame:
    """Draw a deterministic 30-video sample from the 138-video pool.

    Allocation is proportional across populated transition-bit strata. Collapsed
    country is enforced as an exact balancing quota, while ICG and robotic are
    minimum-inclusion floors. A clear error is raised if the constraints cannot
    be satisfied.
    """

    pool = add_collapsed_country(compute_transition_bits(df))
    if len(pool) < n:
        raise ValueError(f"Cannot draw n={n} from pool of {len(pool)}")
    if int((pool["icg"].astype(str) == "1").sum()) < icg_floor:
        raise ValueError("ICG floor exceeds available pool positives")
    if int((pool["robotic"].astype(str) == "1").sum()) < robotic_floor:
        raise ValueError("Robotic floor exceeds available pool positives")

    country_quotas = _country_quotas(pool, n) if country_balance else None
    if country_balance:
        stratum_lower = (
            pool.groupby("stratum")["country_collapsed"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        stratum_quotas = _largest_remainder_quota_with_lower(
            pool["stratum"].value_counts(),
            n,
            stratum_lower,
        )
    else:
        stratum_quotas = _largest_remainder_quota(pool["stratum"].value_counts(), n, min_per_nonzero=1)
    if country_balance:
        joint_counts = pool.groupby(["stratum", "country_collapsed"]).size()
        if len(joint_counts) <= n:
            joint_quotas = _joint_quotas_with_margins(pool, stratum_quotas, country_quotas or {})
        else:
            joint_quotas = None
    else:
        joint_quotas = None
    rng = np.random.default_rng(seed)

    for _ in range(50000):
        parts = []
        if joint_quotas is not None:
            for (stratum, country), want in joint_quotas.items():
                candidates = pool.loc[
                    (pool["stratum"] == stratum)
                    & (pool["country_collapsed"] == country)
                ]
                if len(candidates) < want:
                    raise ValueError(
                        f"Joint cell {(stratum, country)} has {len(candidates)} videos, quota {want}"
                    )
                idx = rng.choice(candidates.index.to_numpy(), size=want, replace=False)
                parts.append(pool.loc[idx])
        else:
            for stratum, want in stratum_quotas.items():
                candidates = pool.loc[pool["stratum"] == stratum]
                if len(candidates) < want:
                    raise ValueError(f"Stratum {stratum} has {len(candidates)} videos, quota {want}")
                idx = rng.choice(candidates.index.to_numpy(), size=want, replace=False)
                parts.append(pool.loc[idx])
        sample = pd.concat(parts, axis=0).sort_values("video_name").reset_index(drop=True)
        stratum_ok = sample["stratum"].value_counts().to_dict() == stratum_quotas
        if stratum_ok and _check_constraints(sample, country_quotas, icg_floor, robotic_floor):
            sample = sample.copy()
            sample["sample_seed"] = int(seed)
            return sample

    raise ValueError(
        "Could not satisfy stratum quotas, country balance, and ICG/robotic floors "
        f"within n={n}; n may be too small."
    )


def compute_poststrat_weights(sample_df: pd.DataFrame, pool_138: pd.DataFrame) -> pd.Series:
    """Return joint post-stratification weights on stratum x collapsed country.

    Weights sum to ``len(pool_138)``. The returned Series has attributes
    ``design_effect`` and ``weight_cells`` for callers that need diagnostics.
    """

    sample = add_collapsed_country(compute_transition_bits(sample_df))
    pool = add_collapsed_country(compute_transition_bits(pool_138))
    keys = ["stratum", "country_collapsed"]
    pool_counts = pool.groupby(keys).size().rename("pool_n")
    sample_counts = sample.groupby(keys).size().rename("sample_n")
    cells = pd.concat([pool_counts, sample_counts], axis=1).fillna(0)
    missing = cells[(cells["pool_n"] > 0) & (cells["sample_n"] == 0)]
    if not missing.empty:
        raise ValueError(f"Sample has no videos for post-strat cells: {list(missing.index)}")
    weight_map = (cells["pool_n"] / cells["sample_n"]).to_dict()
    weights = sample.set_index(keys).index.map(weight_map)
    series = pd.Series(weights, index=sample.index, name="weight", dtype=float)
    cv = float(series.std(ddof=0) / series.mean()) if float(series.mean()) else 0.0
    series.attrs["design_effect"] = 1.0 + cv * cv
    series.attrs["weight_cells"] = cells.reset_index()
    return series


def design_effect(weights: Sequence[float]) -> float:
    """Compute Kish-style design effect ``1 + CV^2`` for weights."""

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0 or arr.mean() == 0:
        return 1.0
    return float(1.0 + (arr.std(ddof=0) / arr.mean()) ** 2)


def summarize_marginals(
    sample_df: pd.DataFrame,
    pool_138: pd.DataFrame,
    pool_168: pd.DataFrame,
) -> pd.DataFrame:
    """Compare sampled 30 vs 138 pool vs 168 eligible for requested marginals."""

    cohorts = {
        "sample_30": add_collapsed_country(compute_transition_bits(sample_df)),
        "pool_138": add_collapsed_country(compute_transition_bits(pool_138)),
        "pool_168": add_collapsed_country(compute_transition_bits(pool_168)),
    }
    rows: list[dict] = []
    fields = [
        "stratum",
        "country_collapsed",
        "country",
        "ioc",
        "icg",
        "robotic",
        "device",
    ]
    for field in fields:
        values = sorted({str(v) for df in cohorts.values() for v in df[field].dropna().astype(str)})
        for value in values:
            row = {"field": field, "value": value}
            for name, df in cohorts.items():
                count = int((df[field].astype(str) == value).sum())
                row[f"{name}_n"] = count
                row[f"{name}_pct"] = count / len(df) if len(df) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def per_cell_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts for transition-bit stratum and stratum x collapsed country."""

    data = add_collapsed_country(compute_transition_bits(df))
    rows: list[dict] = []
    for stratum, count in data["stratum"].value_counts().sort_index().items():
        rows.append({"table": "stratum", "stratum": stratum, "country_collapsed": "", "n": int(count)})
    grouped = data.groupby(["stratum", "country_collapsed"]).size().reset_index(name="n")
    for row in grouped.to_dict("records"):
        rows.append({"table": "stratum_x_country", **row})
    return pd.DataFrame(rows)
