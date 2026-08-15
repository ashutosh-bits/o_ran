"""Leakage-resistant risk models for the O-RAN trace study.

This module intentionally stops at producing a calibrated risk stream.  It does
not choose action thresholds or report threshold-dependent detector metrics;
those belong to the separately tuned containment controller and evaluation
layers.

The public API makes the experiment's separation rules explicit:

* features must be selected from an explicit, metadata-safe allowlist;
* imputation and scaling are fit exactly once, on the named training partition;
* optional Platt calibration uses a distinct named partition; and
* a controller-tuning partition can be registered only when it is distinct from
  both model fitting and probability calibration.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEED = 1729
MODEL_BUNDLE_FORMAT_VERSION = 1
PRIMARY_MODEL_KIND = "logistic"
SENSITIVITY_MODEL_KIND = "hist_gradient_boosting"
MODEL_KINDS = (PRIMARY_MODEL_KIND, SENSITIVITY_MODEL_KIND)

RISK_RAW_COLUMN = "risk_raw"
RISK_SCORE_COLUMN = "risk_score"

# Only causally observed, non-constant radio/MAC KPIs are eligible by default.
# This list is deliberately explicit: a new numeric column never becomes a model
# feature merely because it appeared in a later export.
DEFAULT_KPI_FEATURES: tuple[str, ...] = (
    "mac_dl_cqi",
    "mac_dl_mcs",
    "mac_dl_brate",
    "mac_dl_ok",
    "mac_dl_nok",
    "phy_ul_pusch_sinr",
    "phy_ul_pucch_sinr",
    "phy_ul_mcs",
    "mac_ul_brate",
    "mac_ul_ok",
    "mac_ul_nok",
    "mac_ul_bsr",
    "mac_nof_tti",
    "mac_dl_buffer",
    "mac_phr",
    "mac_dl_cqi_offset",
    "mac_ul_snr_offset",
    "phy_ul_pusch_rssi",
    "phy_ul_pucch_rssi",
    "phy_ul_pucch_ni",
    "phy_ul_turbo_iters",
    "phy_ul_n_samples",
    "phy_ul_n_samples_pucch",
    "phy_dl_mcs",
    "phy_dl_n_samples",
    "samples_in_epoch",
)

# ``rf_error`` is included because the locked study protocol explicitly excludes
# it, not because every field called "error" would necessarily be metadata.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "mac_rnti",
        "ue_ident",
        "id_ue",
        "timestamp",
        "event_time_s",
        "decision_time_s",
        "epoch_start_s",
        "first_event_time_s",
        "last_event_time_s",
        "trace_block_id",
        "rnti_lease_sequence",
        "rnti_lease_id",
        "mob_pattern",
        "mobility",
        "label",
        "label_id",
        "is_attack",
        "is_attack_epoch",
        "labels_in_epoch",
        "target_observed",
        "partition",
        "split",
        "_source_row",
        "rf_error",
        "mac_pci",
        "mac_cc_idx",
    }
)

_FORBIDDEN_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:"
    r"rnti|ue_ident|id_ue|imsi|imei|subscriber|device_id|"
    r"timestamp|event_time|decision_time|epoch_start|"
    r"mob_pattern|mobility|label|target|attack|"
    r"trace_block|lease_id|partition|split|source_row"
    r")(?:_|$)",
    flags=re.IGNORECASE,
)


class FeaturePolicyError(ValueError):
    """Raised when a requested feature violates the frozen feature policy."""


class PartitionLeakageError(ValueError):
    """Raised when fitting, calibration, and tuning partitions overlap."""


class BundleFormatError(ValueError):
    """Raised when a persisted object is not a compatible model bundle."""


@dataclass(frozen=True, slots=True)
class WeightingSummary:
    """Compact audit record for weights used in one fit."""

    components: tuple[str, ...]
    rows: int
    minimum: float
    maximum: float
    mean: float
    effective_sample_size: float


@dataclass(slots=True)
class CausalPreprocessor:
    """Median imputation and scaling that can be fit only once.

    A partition name is mandatory and recorded with the fitted statistics.  The
    one-shot guard makes an accidental refit on calibration or test data fail
    loudly instead of silently changing the score function.
    """

    pipeline: Pipeline = field(
        default_factory=lambda: Pipeline(
            steps=(
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
            )
        )
    )
    training_partition: str | None = None
    fitted_rows: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.training_partition is not None

    @property
    def imputer(self) -> SimpleImputer:
        return self.pipeline.named_steps["imputer"]

    @property
    def scaler(self) -> StandardScaler:
        return self.pipeline.named_steps["scaler"]

    def fit(self, matrix: np.ndarray, *, partition: str) -> "CausalPreprocessor":
        if self.is_fitted:
            raise PartitionLeakageError(
                "preprocessing is already fit on partition "
                f"{self.training_partition!r}; refitting is forbidden"
            )
        partition_name = _partition_name(partition, "training partition")
        array = _two_dimensional_matrix(matrix)
        if array.shape[0] == 0:
            raise ValueError("training matrix must be non-empty")
        self.pipeline.fit(array)
        self.training_partition = partition_name
        self.fitted_rows = int(array.shape[0])
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise PartitionLeakageError("preprocessing must be fit before transform")
        return np.asarray(
            self.pipeline.transform(_two_dimensional_matrix(matrix)), dtype=float
        )


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """One-dimensional logistic calibration fit to held-out raw scores."""

    estimator: LogisticRegression
    partition: str
    seed: int
    weighting: WeightingSummary

    def predict(self, raw_score: Sequence[float] | np.ndarray) -> np.ndarray:
        score = _one_dimensional_float(raw_score, "raw_score")
        return _positive_probability(self.estimator, score.reshape(-1, 1))


@dataclass(frozen=True, slots=True)
class RiskModelBundle:
    """Persistable fitted score model, preprocessing, and partition audit."""

    model_kind: Literal["logistic", "hist_gradient_boosting"]
    feature_columns: tuple[str, ...]
    feature_allowlist: tuple[str, ...]
    preprocessor: CausalPreprocessor
    estimator: LogisticRegression | HistGradientBoostingClassifier
    seed: int
    training_partition: str
    training_weighting: WeightingSummary
    calibrator: PlattCalibrator | None = None
    controller_tuning_partition: str | None = None
    format_version: int = MODEL_BUNDLE_FORMAT_VERSION
    library_versions: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: (
            ("numpy", np.__version__),
            ("polars", pl.__version__),
            ("scikit-learn", sklearn.__version__),
        )
    )

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator is not None

    @property
    def calibration_partition(self) -> str | None:
        return None if self.calibrator is None else self.calibrator.partition

    def raw_score(self, frame: pl.DataFrame) -> np.ndarray:
        """Return the base estimator's real-valued score in input row order."""

        columns = assert_metadata_safe_feature_allowlist(
            frame,
            self.feature_columns,
            allowed_feature_universe=self.feature_allowlist,
        )
        transformed = self.preprocessor.transform(_feature_matrix(frame, columns))
        if hasattr(self.estimator, "decision_function"):
            result = np.asarray(self.estimator.decision_function(transformed), dtype=float)
            if result.ndim == 2:
                positive_index = _positive_class_index(self.estimator)
                result = result[:, positive_index]
        else:
            probability = _positive_probability(self.estimator, transformed)
            probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
            result = np.log(probability / (1.0 - probability))
        result = np.asarray(result, dtype=float).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError("base estimator produced a non-finite raw score")
        return result

    def predict_risk(self, frame: pl.DataFrame) -> np.ndarray:
        """Return calibrated probabilities, or base probabilities if uncalibrated."""

        raw = self.raw_score(frame)
        if self.calibrator is not None:
            probability = self.calibrator.predict(raw)
        else:
            probability = _stable_sigmoid(raw)
        if not np.isfinite(probability).all():
            raise ValueError("risk model produced a non-finite probability")
        return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)

    def score_frame(
        self,
        frame: pl.DataFrame,
        *,
        passthrough_columns: Sequence[str] = (),
        controller_tuning_partition: str | None = None,
    ) -> pl.DataFrame:
        """Return selected row metadata plus raw and probability scores.

        Passthrough fields are copied only after scoring and therefore never enter
        the feature matrix.  Supplying ``controller_tuning_partition`` activates
        the calibration-separation assertion.
        """

        if controller_tuning_partition is not None:
            self.assert_controller_tuning_partition(controller_tuning_partition)
        return score_frame(self, frame, passthrough_columns=passthrough_columns)

    def assert_controller_tuning_partition(self, partition: str) -> None:
        """Fail unless ``partition`` is reserved exclusively for controller tuning."""

        candidate = _partition_name(partition, "controller tuning partition")
        occupied = {self.training_partition}
        if self.calibration_partition is not None:
            occupied.add(self.calibration_partition)
        if candidate in occupied:
            raise PartitionLeakageError(
                "controller tuning must use a partition distinct from model fitting "
                "and Platt calibration"
            )
        if (
            self.controller_tuning_partition is not None
            and candidate != self.controller_tuning_partition
        ):
            raise PartitionLeakageError(
                "controller tuning partition is already registered as "
                f"{self.controller_tuning_partition!r}"
            )


def _partition_name(value: str, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty string")
    return value.strip()


def _validate_column_names(columns: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of column names, not a string")
    result = tuple(columns)
    if not result:
        raise FeaturePolicyError(f"{name} must not be empty")
    if any(not isinstance(column, str) or not column for column in result):
        raise TypeError(f"{name} must contain non-empty strings")
    duplicates = sorted({column for column in result if result.count(column) > 1})
    if duplicates:
        raise FeaturePolicyError(f"duplicate feature columns: {duplicates}")
    return result


def _unsafe_feature_name(column: str) -> bool:
    normalized = column.strip().lower()
    return (
        normalized in FORBIDDEN_FEATURE_COLUMNS
        or normalized.startswith("_")
        or _FORBIDDEN_NAME_PATTERN.search(normalized) is not None
    )


def assert_metadata_safe_feature_allowlist(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    allowed_feature_universe: Sequence[str] = DEFAULT_KPI_FEATURES,
) -> tuple[str, ...]:
    """Validate an explicit feature list and return its stable column order.

    Metadata-like names are rejected even if a caller accidentally places them
    in ``allowed_feature_universe``.  Numeric columns outside that universe are
    also rejected, preventing schema drift from silently changing a model.
    """

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")
    features = _validate_column_names(feature_columns, "feature_columns")
    universe = _validate_column_names(
        allowed_feature_universe, "allowed_feature_universe"
    )
    unsafe_universe = sorted(column for column in universe if _unsafe_feature_name(column))
    if unsafe_universe:
        raise FeaturePolicyError(
            "feature allowlist itself contains identifier, context, time, target, "
            f"or protocol-excluded columns: {unsafe_universe}"
        )
    unsafe = sorted(column for column in features if _unsafe_feature_name(column))
    if unsafe:
        raise FeaturePolicyError(
            "identifier, context, time, target, or protocol-excluded features are "
            f"forbidden: {unsafe}"
        )
    unknown = sorted(set(features) - set(universe))
    if unknown:
        raise FeaturePolicyError(
            f"features are absent from the explicit allowlist: {unknown}"
        )
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise FeaturePolicyError(f"feature columns are missing from frame: {missing}")
    non_numeric = sorted(
        column for column in features if not frame.schema[column].is_numeric()
    )
    if non_numeric:
        raise FeaturePolicyError(f"feature columns must be numeric: {non_numeric}")
    return features


def _feature_matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = frame.select(
        pl.col(column).cast(pl.Float64, strict=True).alias(column) for column in columns
    ).to_numpy()
    matrix = np.asarray(matrix, dtype=float)
    if np.isinf(matrix).any():
        locations = np.argwhere(np.isinf(matrix))
        first_row, first_column = locations[0]
        raise ValueError(
            "feature matrix contains infinity at row "
            f"{int(first_row)}, column {columns[int(first_column)]!r}"
        )
    return matrix


def _two_dimensional_matrix(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    if np.isinf(result).any():
        raise ValueError("feature matrix must not contain infinity")
    return result


def _one_dimensional_float(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(result) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _resolve_vector(frame: pl.DataFrame, value: str | Sequence[Any], name: str) -> np.ndarray:
    if isinstance(value, str):
        if value not in frame.columns:
            raise ValueError(f"{name} column {value!r} is missing from frame")
        result = frame.get_column(value).to_numpy()
    else:
        result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(result) != frame.height:
        raise ValueError(
            f"{name} has {len(result)} rows but frame has {frame.height} rows"
        )
    return result


def _binary_target(value: Any, name: str = "target") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be encoded as binary 0/1 values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return numeric.astype(np.int8)


def _timestamps_seconds(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    if array.dtype.kind == "M":
        if np.isnat(array).any():
            raise ValueError("timestamps must not contain NaT")
        result = array.astype("datetime64[ns]").astype(np.int64) / 1e9
    elif array.dtype.kind in "iuf":
        result = array.astype(float, copy=False)
    else:
        converted: list[float] = []
        for value in array:
            if hasattr(value, "timestamp"):
                converted.append(float(value.timestamp()))
            else:
                try:
                    timestamp = np.datetime64(value, "ns")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid timestamp value {value!r}") from exc
                if np.isnat(timestamp):
                    raise ValueError("timestamps must not contain NaT")
                converted.append(float(timestamp.astype(np.int64)) / 1e9)
        result = np.asarray(converted, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("timestamps must contain only finite values")
    return result


def _group_codes(groups: Any, rows: int) -> np.ndarray:
    if groups is None:
        return np.zeros(rows, dtype=np.int64)
    array = np.asarray(groups)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError("groups must be one-dimensional and match timestamps")
    encoded = np.empty(rows, dtype=np.int64)
    keys: dict[tuple[str, str], int] = {}
    for index, value in enumerate(array):
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            raise ValueError("groups must not contain null values")
        key = (type(value).__qualname__, repr(value))
        if key not in keys:
            keys[key] = len(keys)
        encoded[index] = keys[key]
    return encoded


def elapsed_time_sample_weights(
    timestamps: Sequence[Any] | np.ndarray,
    *,
    groups: Sequence[Any] | np.ndarray | None = None,
    max_gap: float | None = None,
    min_interval: float = 0.0,
) -> np.ndarray:
    """Return mean-one dwell-time weights without crossing group boundaries.

    Each observation receives the time until the next observation in the same
    group.  A group's last observation uses that group's median positive interval
    (or the global median, then one second, for singleton/duplicate-only groups).
    This prevents a gap between independent trace blocks or RNTI leases from
    becoming sample weight.  The caller should therefore provide the relevant
    lifecycle group whenever multiple streams share a frame.
    """

    times = _timestamps_seconds(timestamps)
    if not math.isfinite(min_interval) or min_interval < 0:
        raise ValueError("min_interval must be finite and non-negative")
    if max_gap is not None:
        if not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be finite and positive")
        if min_interval > max_gap:
            raise ValueError("min_interval must not exceed max_gap")
    codes = _group_codes(groups, len(times))

    group_indices = [np.flatnonzero(codes == code) for code in range(codes.max() + 1)]
    positive_intervals: list[np.ndarray] = []
    group_differences: list[np.ndarray] = []
    for indices in group_indices:
        differences = np.diff(times[indices])
        if np.any(differences < 0):
            raise ValueError("timestamps must be non-decreasing within every group")
        group_differences.append(differences)
        positive = differences[differences > 0]
        if len(positive):
            positive_intervals.append(positive)
    global_default = (
        float(np.median(np.concatenate(positive_intervals)))
        if positive_intervals
        else 1.0
    )

    weights = np.empty(len(times), dtype=float)
    for indices, differences in zip(group_indices, group_differences, strict=True):
        local_positive = differences[differences > 0]
        terminal = (
            float(np.median(local_positive)) if len(local_positive) else global_default
        )
        if len(indices) > 1:
            weights[indices[:-1]] = differences
        weights[indices[-1]] = terminal
    if max_gap is not None:
        weights = np.minimum(weights, float(max_gap))
    if min_interval:
        weights = np.maximum(weights, float(min_interval))
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("elapsed-time weights have no positive finite mass")
    return weights / weights.mean()


def compose_sample_weights(
    target: Sequence[int] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    timestamps: Sequence[Any] | np.ndarray | None = None,
    groups: Sequence[Any] | np.ndarray | None = None,
    balance_classes: bool = False,
    max_time_gap: float | None = None,
    min_time_interval: float = 0.0,
) -> tuple[np.ndarray, WeightingSummary]:
    """Combine caller weights, dwell-time weights, and optional class balancing."""

    y = _binary_target(target)
    weights = np.ones(len(y), dtype=float)
    components: list[str] = []
    if sample_weight is not None:
        supplied = _one_dimensional_float(sample_weight, "sample_weight")
        if len(supplied) != len(y):
            raise ValueError("sample_weight must have the same length as target")
        if np.any(supplied < 0) or supplied.sum() <= 0:
            raise ValueError("sample_weight must be non-negative with positive mass")
        weights *= supplied
        components.append("sample")
    if timestamps is not None:
        if len(np.asarray(timestamps)) != len(y):
            raise ValueError("timestamps must have the same length as target")
        weights *= elapsed_time_sample_weights(
            timestamps,
            groups=groups,
            max_gap=max_time_gap,
            min_interval=min_time_interval,
        )
        components.append("elapsed_time")
    elif groups is not None:
        raise ValueError("groups are meaningful only when timestamps are supplied")
    if balance_classes:
        present = np.unique(y)
        if len(present) != 2:
            raise ValueError("class balancing requires both binary target classes")
        total = float(weights.sum())
        for target_class in (0, 1):
            mask = y == target_class
            class_mass = float(weights[mask].sum())
            if class_mass <= 0:
                raise ValueError(f"target class {target_class} has zero weight")
            weights[mask] *= total / (2.0 * class_mass)
        components.append("balanced_classes")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("combined sample weights must be finite and non-negative")
    weights *= len(weights) / weights.sum()
    squared_sum = float(np.square(weights).sum())
    effective = float(weights.sum() ** 2 / squared_sum) if squared_sum else 0.0
    summary = WeightingSummary(
        components=tuple(components) if components else ("uniform",),
        rows=len(weights),
        minimum=float(weights.min()),
        maximum=float(weights.max()),
        mean=float(weights.mean()),
        effective_sample_size=effective,
    )
    return weights, summary


def _positive_class_index(estimator: Any) -> int:
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("fitted estimator does not expose binary positive class 1")
    return int(matches[0])


def _positive_probability(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("estimator predict_proba returned an invalid shape")
    return probabilities[:, _positive_class_index(estimator)]


def _stable_sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    result = np.empty_like(score)
    nonnegative = score >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-score[nonnegative]))
    exponential = np.exp(score[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _make_estimator(
    model_kind: str,
    *,
    seed: int,
    regularization_c: float,
    hgb_max_depth: int,
    hgb_max_leaf_nodes: int,
    hgb_learning_rate: float,
    hgb_max_iter: int,
    hgb_l2_regularization: float,
) -> LogisticRegression | HistGradientBoostingClassifier:
    if model_kind == PRIMARY_MODEL_KIND:
        if not math.isfinite(regularization_c) or regularization_c <= 0:
            raise ValueError("regularization_c must be finite and positive")
        return LogisticRegression(
            C=float(regularization_c),
            solver="lbfgs",
            max_iter=2_000,
            tol=1e-8,
            random_state=int(seed),
        )
    if model_kind == SENSITIVITY_MODEL_KIND:
        if not isinstance(hgb_max_depth, int) or not 1 <= hgb_max_depth <= 3:
            raise ValueError("sensitivity HGB max_depth must be in [1, 3]")
        if not isinstance(hgb_max_leaf_nodes, int) or not 2 <= hgb_max_leaf_nodes <= 15:
            raise ValueError("sensitivity HGB max_leaf_nodes must be in [2, 15]")
        if not math.isfinite(hgb_learning_rate) or hgb_learning_rate <= 0:
            raise ValueError("hgb_learning_rate must be finite and positive")
        if not isinstance(hgb_max_iter, int) or hgb_max_iter < 1:
            raise ValueError("hgb_max_iter must be a positive integer")
        if not math.isfinite(hgb_l2_regularization) or hgb_l2_regularization < 0:
            raise ValueError("hgb_l2_regularization must be finite and non-negative")
        return HistGradientBoostingClassifier(
            learning_rate=float(hgb_learning_rate),
            max_iter=hgb_max_iter,
            max_leaf_nodes=hgb_max_leaf_nodes,
            max_depth=hgb_max_depth,
            min_samples_leaf=20,
            l2_regularization=float(hgb_l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        )
    raise ValueError(f"model_kind must be one of {MODEL_KINDS}; got {model_kind!r}")


def fit_risk_model(
    frame: pl.DataFrame,
    target: str | Sequence[int] | np.ndarray,
    *,
    feature_columns: Sequence[str] | None = None,
    allowed_feature_universe: Sequence[str] = DEFAULT_KPI_FEATURES,
    model_kind: Literal["logistic", "hist_gradient_boosting"] = PRIMARY_MODEL_KIND,
    training_partition: str = "train",
    sample_weight: str | Sequence[float] | np.ndarray | None = None,
    timestamps: str | Sequence[Any] | np.ndarray | None = None,
    groups: str | Sequence[Any] | np.ndarray | None = None,
    balance_classes: bool = False,
    max_time_gap: float | None = None,
    min_time_interval: float = 0.0,
    seed: int = DEFAULT_SEED,
    regularization_c: float = 1.0,
    hgb_max_depth: int = 3,
    hgb_max_leaf_nodes: int = 7,
    hgb_learning_rate: float = 0.05,
    hgb_max_iter: int = 150,
    hgb_l2_regularization: float = 1.0,
) -> RiskModelBundle:
    """Fit the primary regularized logistic model or shallow HGB sensitivity.

    ``sample_weight``, ``timestamps``, and ``groups`` may name frame columns or
    provide arrays.  They are resolved separately and never appended to the
    feature matrix.  If timestamps are requested without explicit groups, the
    function uses ``rnti_lease_id`` and then ``trace_block_id`` when present.
    """

    partition = _partition_name(training_partition, "training partition")
    universe = _validate_column_names(
        allowed_feature_universe, "allowed_feature_universe"
    )
    selected = (
        tuple(column for column in universe if column in frame.columns)
        if feature_columns is None
        else tuple(feature_columns)
    )
    selected = assert_metadata_safe_feature_allowlist(
        frame, selected, allowed_feature_universe=universe
    )
    y = _binary_target(_resolve_vector(frame, target, "target"))
    if len(np.unique(y)) != 2:
        raise ValueError("model fitting requires both binary target classes")

    supplied_weight = (
        None
        if sample_weight is None
        else _resolve_vector(frame, sample_weight, "sample_weight")
    )
    supplied_timestamps = (
        None if timestamps is None else _resolve_vector(frame, timestamps, "timestamps")
    )
    resolved_groups = groups
    if supplied_timestamps is not None and groups is None:
        for candidate in ("rnti_lease_id", "trace_block_id"):
            if candidate in frame.columns:
                resolved_groups = candidate
                break
    supplied_groups = (
        None if resolved_groups is None else _resolve_vector(frame, resolved_groups, "groups")
    )
    weights, weighting_summary = compose_sample_weights(
        y,
        sample_weight=supplied_weight,
        timestamps=supplied_timestamps,
        groups=supplied_groups,
        balance_classes=balance_classes,
        max_time_gap=max_time_gap,
        min_time_interval=min_time_interval,
    )

    preprocessor = CausalPreprocessor()
    transformed = preprocessor.fit(
        _feature_matrix(frame, selected), partition=partition
    ).transform(_feature_matrix(frame, selected))
    estimator = _make_estimator(
        model_kind,
        seed=seed,
        regularization_c=regularization_c,
        hgb_max_depth=hgb_max_depth,
        hgb_max_leaf_nodes=hgb_max_leaf_nodes,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_iter=hgb_max_iter,
        hgb_l2_regularization=hgb_l2_regularization,
    )
    estimator.fit(transformed, y, sample_weight=weights)
    return RiskModelBundle(
        model_kind=model_kind,
        feature_columns=selected,
        feature_allowlist=universe,
        preprocessor=preprocessor,
        estimator=estimator,
        seed=int(seed),
        training_partition=partition,
        training_weighting=weighting_summary,
    )


def fit_primary_logistic(
    frame: pl.DataFrame,
    target: str | Sequence[int] | np.ndarray,
    **kwargs: Any,
) -> RiskModelBundle:
    """Fit the predeclared primary, L2-regularized logistic risk model."""

    if "model_kind" in kwargs:
        raise TypeError("fit_primary_logistic fixes model_kind='logistic'")
    return fit_risk_model(frame, target, model_kind=PRIMARY_MODEL_KIND, **kwargs)


def fit_shallow_hgb_sensitivity(
    frame: pl.DataFrame,
    target: str | Sequence[int] | np.ndarray,
    **kwargs: Any,
) -> RiskModelBundle:
    """Fit the bounded-depth nonlinear sensitivity model."""

    if "model_kind" in kwargs:
        raise TypeError(
            "fit_shallow_hgb_sensitivity fixes model_kind='hist_gradient_boosting'"
        )
    return fit_risk_model(
        frame, target, model_kind=SENSITIVITY_MODEL_KIND, **kwargs
    )


def fit_platt_calibrator(
    bundle: RiskModelBundle,
    frame: pl.DataFrame,
    target: str | Sequence[int] | np.ndarray,
    *,
    calibration_partition: str,
    controller_tuning_partition: str | None = None,
    sample_weight: str | Sequence[float] | np.ndarray | None = None,
    timestamps: str | Sequence[Any] | np.ndarray | None = None,
    groups: str | Sequence[Any] | np.ndarray | None = None,
    balance_classes: bool = False,
    max_time_gap: float | None = None,
    min_time_interval: float = 0.0,
    seed: int | None = None,
) -> RiskModelBundle:
    """Fit Platt scaling on held-out data without changing base preprocessing.

    Controller tuning is not performed here.  Supplying its future partition is
    optional, but when supplied it is registered and checked for three-way
    separation from the training and calibration partitions.
    """

    if not isinstance(bundle, RiskModelBundle):
        raise TypeError("bundle must be a RiskModelBundle")
    if bundle.calibrator is not None:
        raise PartitionLeakageError("bundle is already calibrated; refitting is forbidden")
    calibration = _partition_name(calibration_partition, "calibration partition")
    if calibration == bundle.training_partition:
        raise PartitionLeakageError(
            "Platt calibration must not reuse the model-fitting partition"
        )
    tuning = None
    if controller_tuning_partition is not None:
        tuning = _partition_name(
            controller_tuning_partition, "controller tuning partition"
        )
        if tuning in {bundle.training_partition, calibration}:
            raise PartitionLeakageError(
                "controller tuning must use a third partition, distinct from model "
                "fitting and Platt calibration"
            )

    y = _binary_target(_resolve_vector(frame, target, "target"))
    if len(np.unique(y)) != 2:
        raise ValueError("Platt calibration requires both binary target classes")
    supplied_weight = (
        None
        if sample_weight is None
        else _resolve_vector(frame, sample_weight, "sample_weight")
    )
    supplied_timestamps = (
        None if timestamps is None else _resolve_vector(frame, timestamps, "timestamps")
    )
    resolved_groups = groups
    if supplied_timestamps is not None and groups is None:
        for candidate in ("rnti_lease_id", "trace_block_id"):
            if candidate in frame.columns:
                resolved_groups = candidate
                break
    supplied_groups = (
        None if resolved_groups is None else _resolve_vector(frame, resolved_groups, "groups")
    )
    weights, weighting_summary = compose_sample_weights(
        y,
        sample_weight=supplied_weight,
        timestamps=supplied_timestamps,
        groups=supplied_groups,
        balance_classes=balance_classes,
        max_time_gap=max_time_gap,
        min_time_interval=min_time_interval,
    )
    calibration_seed = bundle.seed if seed is None else int(seed)
    estimator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-10,
        random_state=calibration_seed,
    )
    estimator.fit(bundle.raw_score(frame).reshape(-1, 1), y, sample_weight=weights)
    calibrator = PlattCalibrator(
        estimator=estimator,
        partition=calibration,
        seed=calibration_seed,
        weighting=weighting_summary,
    )
    return replace(
        bundle,
        calibrator=calibrator,
        controller_tuning_partition=tuning,
    )


def register_controller_tuning_partition(
    bundle: RiskModelBundle, partition: str
) -> RiskModelBundle:
    """Return a bundle with an exclusively reserved controller-tuning partition."""

    bundle.assert_controller_tuning_partition(partition)
    return replace(
        bundle,
        controller_tuning_partition=_partition_name(
            partition, "controller tuning partition"
        ),
    )


def score_frame(
    bundle: RiskModelBundle,
    frame: pl.DataFrame,
    *,
    passthrough_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Create a stable-order Polars score frame without exposing metadata to X."""

    if isinstance(passthrough_columns, (str, bytes)):
        raise TypeError("passthrough_columns must be a sequence, not a string")
    passthrough = tuple(passthrough_columns)
    if len(set(passthrough)) != len(passthrough):
        raise ValueError("passthrough_columns must not contain duplicates")
    missing = sorted(set(passthrough) - set(frame.columns))
    if missing:
        raise ValueError(f"passthrough columns are missing from frame: {missing}")
    collisions = sorted(set(passthrough) & {RISK_RAW_COLUMN, RISK_SCORE_COLUMN})
    if collisions:
        raise ValueError(f"passthrough columns collide with score fields: {collisions}")
    raw = bundle.raw_score(frame)
    probability = (
        bundle.calibrator.predict(raw)
        if bundle.calibrator is not None
        else _stable_sigmoid(raw)
    )
    result = frame.select(passthrough) if passthrough else pl.DataFrame()
    if not passthrough:
        result = pl.DataFrame({RISK_RAW_COLUMN: raw, RISK_SCORE_COLUMN: probability})
    else:
        result = result.with_columns(
            pl.Series(RISK_RAW_COLUMN, raw, dtype=pl.Float64),
            pl.Series(RISK_SCORE_COLUMN, probability, dtype=pl.Float64),
        )
    _validate_score_frame(result)
    return result


def risk_diagnostics(
    target: Sequence[int] | np.ndarray,
    risk_score: Sequence[float] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, int | float]:
    """Return threshold-free diagnostics for a fixed risk stream.

    AUROC, AUPRC, and Brier score characterize the nuisance risk model; they are
    not used here to select a detector threshold or claim the paper's primary
    contribution.  AUROC/AUPRC are ``nan`` for single-class diagnostic subsets.
    """

    y = _binary_target(target)
    probability = _one_dimensional_float(risk_score, "risk_score")
    if len(probability) != len(y):
        raise ValueError("risk_score must have the same length as target")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("risk_score must lie in [0, 1]")
    weights = None
    if sample_weight is not None:
        weights = _one_dimensional_float(sample_weight, "sample_weight")
        if len(weights) != len(y):
            raise ValueError("sample_weight must have the same length as target")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("sample_weight must be non-negative with positive mass")
    prevalence = (
        float(np.mean(y))
        if weights is None
        else float(np.average(y.astype(float), weights=weights))
    )
    if len(np.unique(y)) == 2:
        auroc = float(roc_auc_score(y, probability, sample_weight=weights))
        auprc = float(average_precision_score(y, probability, sample_weight=weights))
    else:
        auroc = math.nan
        auprc = math.nan
    return {
        "n_rows": int(len(y)),
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(brier_score_loss(y, probability, sample_weight=weights)),
    }


def diagnose_model(
    bundle: RiskModelBundle,
    frame: pl.DataFrame,
    target: str | Sequence[int] | np.ndarray,
    *,
    sample_weight: str | Sequence[float] | np.ndarray | None = None,
) -> dict[str, int | float]:
    """Convenience wrapper for threshold-free bundle diagnostics."""

    y = _resolve_vector(frame, target, "target")
    weights = (
        None
        if sample_weight is None
        else _resolve_vector(frame, sample_weight, "sample_weight")
    )
    return risk_diagnostics(y, bundle.predict_risk(frame), sample_weight=weights)


def save_model_bundle(bundle: RiskModelBundle, path: str | os.PathLike[str]) -> Path:
    """Atomically persist a model bundle with joblib."""

    if not isinstance(bundle, RiskModelBundle):
        raise TypeError("bundle must be a RiskModelBundle")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        joblib.dump(bundle, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_model_bundle(path: str | os.PathLike[str]) -> RiskModelBundle:
    """Load and validate a trusted model bundle.

    Joblib uses pickle semantics; callers must not load files from untrusted
    sources.
    """

    loaded = joblib.load(Path(path))
    if not isinstance(loaded, RiskModelBundle):
        raise BundleFormatError("persisted object is not a RiskModelBundle")
    if loaded.format_version != MODEL_BUNDLE_FORMAT_VERSION:
        raise BundleFormatError(
            "unsupported model bundle format version "
            f"{loaded.format_version}; expected {MODEL_BUNDLE_FORMAT_VERSION}"
        )
    if loaded.training_partition != loaded.preprocessor.training_partition:
        raise BundleFormatError("bundle and preprocessor training partitions disagree")
    assert_metadata_safe_feature_allowlist(
        pl.DataFrame(
            {
                column: pl.Series(column, [], dtype=pl.Float64)
                for column in loaded.feature_columns
            }
        ),
        loaded.feature_columns,
        allowed_feature_universe=loaded.feature_allowlist,
    )
    return loaded


def _validate_score_frame(frame: pl.DataFrame) -> None:
    missing = {RISK_RAW_COLUMN, RISK_SCORE_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"score frame is missing columns: {sorted(missing)}")
    for column in (RISK_RAW_COLUMN, RISK_SCORE_COLUMN):
        if not frame.schema[column].is_numeric():
            raise ValueError(f"score column {column!r} must be numeric")
        values = frame.get_column(column).cast(pl.Float64).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"score column {column!r} must contain finite values")
    probability = frame.get_column(RISK_SCORE_COLUMN).cast(pl.Float64).to_numpy()
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{RISK_SCORE_COLUMN} must lie in [0, 1]")


def save_score_frame(frame: pl.DataFrame, path: str | os.PathLike[str]) -> Path:
    """Atomically save a validated score frame as Parquet or CSV."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")
    _validate_score_frame(frame)
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".parquet", ".csv"}:
        raise ValueError("score frame path must end in .parquet or .csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        if suffix == ".parquet":
            frame.write_parquet(temporary)
        else:
            frame.write_csv(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_score_frame(path: str | os.PathLike[str]) -> pl.DataFrame:
    """Load and validate a Parquet or CSV score frame."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(source)
    elif suffix == ".csv":
        frame = pl.read_csv(source)
    else:
        raise ValueError("score frame path must end in .parquet or .csv")
    _validate_score_frame(frame)
    return frame


__all__ = [
    "BundleFormatError",
    "CausalPreprocessor",
    "DEFAULT_KPI_FEATURES",
    "DEFAULT_SEED",
    "FeaturePolicyError",
    "MODEL_BUNDLE_FORMAT_VERSION",
    "PRIMARY_MODEL_KIND",
    "PartitionLeakageError",
    "PlattCalibrator",
    "RISK_RAW_COLUMN",
    "RISK_SCORE_COLUMN",
    "RiskModelBundle",
    "SENSITIVITY_MODEL_KIND",
    "WeightingSummary",
    "assert_metadata_safe_feature_allowlist",
    "compose_sample_weights",
    "diagnose_model",
    "elapsed_time_sample_weights",
    "fit_platt_calibrator",
    "fit_primary_logistic",
    "fit_risk_model",
    "fit_shallow_hgb_sensitivity",
    "load_model_bundle",
    "load_score_frame",
    "register_controller_tuning_partition",
    "risk_diagnostics",
    "save_model_bundle",
    "save_score_frame",
    "score_frame",
]
