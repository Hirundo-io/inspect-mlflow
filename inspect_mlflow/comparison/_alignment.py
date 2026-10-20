"""Sample alignment across evaluation runs.

Aligns samples between two eval logs by (id, epoch) key, matching
Inspect's own retry logic at _eval/task/run.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

from inspect_ai.log._log import EvalLog, EvalSample

logger = getLogger(__name__)


@dataclass
class AlignedSample:
    """A pair of samples aligned by (id, epoch)."""

    id: int | str
    epoch: int
    baseline: EvalSample | None
    candidate: EvalSample | None


def _sample_key(sample_id: int | str) -> int | str:
    """Normalize canonical ASCII integer strings without collapsing other IDs."""
    if (
        isinstance(sample_id, str)
        and sample_id.isascii()
        and sample_id.isdecimal()
        and (sample_id == "0" or not sample_id.startswith("0"))
    ):
        try:
            return int(sample_id)
        except ValueError:
            # Python can limit decimal conversion length; long string IDs remain valid.
            return sample_id
    return sample_id


def _index_samples(
    samples: list[EvalSample], log_name: str
) -> dict[tuple[int | str, int], EvalSample]:
    """Index samples, rejecting duplicate or ambiguous normalized identities."""
    indexed: dict[tuple[int | str, int], EvalSample] = {}
    for sample in samples:
        key = (_sample_key(sample.id), sample.epoch)
        if key in indexed:
            raise ValueError(
                f"Duplicate or ambiguous sample IDs in {log_name} log at epoch {sample.epoch}"
            )
        indexed[key] = sample
    return indexed


def align_samples(
    baseline_log: EvalLog,
    candidate_log: EvalLog,
) -> list[AlignedSample]:
    """Align samples from two eval logs by (id, epoch).

    Canonical ASCII integer strings match integer IDs across logs. Other
    string IDs, including zero-padded numbers, retain their identity. Samples
    present in only one log are included with None for the missing side.

    Args:
        baseline_log: The baseline evaluation log.
        candidate_log: The candidate evaluation log.

    Returns:
        List of AlignedSample pairs, ordered by (id, epoch).

    Raises:
        ValueError: If either log contains duplicate or ambiguous normalized
            sample IDs within an epoch.
    """
    baseline_samples = baseline_log.samples or []
    candidate_samples = candidate_log.samples or []

    baseline_map = _index_samples(baseline_samples, "baseline")
    candidate_map = _index_samples(candidate_samples, "candidate")

    all_keys = sorted(
        set(baseline_map.keys()) | set(candidate_map.keys()),
        key=lambda k: (
            (0, k[0], "") if isinstance(k[0], int) else (1, 0, str(k[0])),
            k[1],
        ),
    )

    aligned: list[AlignedSample] = []
    for key in all_keys:
        sample_id, epoch = key
        aligned.append(
            AlignedSample(
                id=sample_id,
                epoch=epoch,
                baseline=baseline_map.get(key),
                candidate=candidate_map.get(key),
            )
        )

    baseline_only = sum(1 for a in aligned if a.candidate is None)
    candidate_only = sum(1 for a in aligned if a.baseline is None)
    matched = sum(1 for a in aligned if a.baseline is not None and a.candidate is not None)

    logger.info(
        "Aligned %d samples: %d matched, %d baseline-only, %d candidate-only",
        len(aligned),
        matched,
        baseline_only,
        candidate_only,
    )

    return aligned
