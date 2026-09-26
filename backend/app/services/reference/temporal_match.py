"""
Temporal Matching Component
===========================
Evaluates acquisition dates of target scene and reference candidate.
Computes temporal delta, classifies compatibility status, and generates scientific caveats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re


@dataclass
class TemporalMatchResult:
    """Result of temporal proximity analysis."""
    temporal_difference_days: int | None
    temporal_match_status: str  # EXACT | CLOSE | ACCEPTABLE_WITH_CAVEAT | LARGE_DIFFERENCE | UNKNOWN
    target_date: str | None
    reference_date: str | None
    is_acceptable: bool
    notes: list[str]


class TemporalMatcher:
    """Classifies temporal alignment between target acquisition and reference dataset."""

    def __init__(
        self,
        exact_days: int = 0,
        close_days: int = 7,
        acceptable_with_caveat_days: int = 30,
        max_allowed_days: int = 365,
    ) -> None:
        self.exact_days = exact_days
        self.close_days = close_days
        self.acceptable_with_caveat_days = acceptable_with_caveat_days
        self.max_allowed_days = max_allowed_days

    def parse_date(self, date_val: Any) -> datetime | None:
        """Parse various date string formats or datetime objects into a timezone-naive UTC date."""
        if not date_val:
            return None
        from datetime import date as dt_date
        if isinstance(date_val, datetime):
            if date_val.tzinfo is not None:
                date_val = date_val.astimezone(timezone.utc).replace(tzinfo=None)
            return date_val
        if isinstance(date_val, dt_date):
            return datetime(date_val.year, date_val.month, date_val.day)

        s = str(date_val).strip()
        # ISO format anywhere in string: 2023-06-15 or 2023-06-15T12:00:00Z
        iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if iso_match:
            try:
                y, m, d = map(int, iso_match.groups())
                if 2000 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                    return datetime(y, m, d)
            except Exception:
                pass

        # Compact format: 20230615 (e.g. INDIA_REF_CHENNAI_20230615 or S2A_..._20230615T...)
        compact_match = re.search(r"(?:^|[^0-9])(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:[^0-9]|$)", s)
        if compact_match:
            try:
                y, m, d = map(int, compact_match.groups())
                return datetime(y, m, d)
            except Exception:
                pass

        return None

    def evaluate(
        self,
        target_date: str | datetime | None,
        reference_date: str | datetime | None,
    ) -> TemporalMatchResult:
        """Compute delta in days and classify temporal status."""
        dt_target = self.parse_date(target_date)
        dt_ref = self.parse_date(reference_date)

        target_str = dt_target.strftime("%Y-%m-%d") if dt_target else (str(target_date) if target_date else None)
        ref_str = dt_ref.strftime("%Y-%m-%d") if dt_ref else (str(reference_date) if reference_date else None)

        if dt_target is None or dt_ref is None:
            return TemporalMatchResult(
                temporal_difference_days=None,
                temporal_match_status="UNKNOWN",
                target_date=target_str,
                reference_date=ref_str,
                is_acceptable=True,  # Allowed to proceed under UNKNOWN temporal provenance
                notes=["One or both acquisition dates unavailable; temporal difference is UNKNOWN."],
            )

        delta_days = abs((dt_target.date() - dt_ref.date()).days)
        notes: list[str] = [f"Temporal separation: {delta_days} day(s)."]

        if delta_days <= self.exact_days:
            status = "EXACT"
            acceptable = True
            notes.append("Exact temporal match between target and reference imagery.")
        elif delta_days <= self.close_days:
            status = "CLOSE"
            acceptable = True
            notes.append("High temporal proximity (<= 7 days). Phenological state is highly consistent.")
        elif delta_days <= self.acceptable_with_caveat_days:
            status = "ACCEPTABLE_WITH_CAVEAT"
            acceptable = True
            notes.append(
                f"Moderate temporal separation ({delta_days} days). "
                "Seasonal phenology or solar angle variations may introduce minor discrepancies."
            )
        elif delta_days <= self.max_allowed_days:
            status = "LARGE_DIFFERENCE"
            acceptable = False
            notes.append(
                f"Substantial temporal mismatch ({delta_days} days > {self.acceptable_with_caveat_days} days). "
                "Land cover, vegetation phenology, or infrastructure changes invalidate strict pixel reconstruction comparison."
            )
        else:
            status = "LARGE_DIFFERENCE"
            acceptable = False
            notes.append(
                f"Extreme temporal separation ({delta_days} days). Reference rejected as temporally incompatible."
            )

        return TemporalMatchResult(
            temporal_difference_days=delta_days,
            temporal_match_status=status,
            target_date=target_str,
            reference_date=ref_str,
            is_acceptable=acceptable,
            notes=notes,
        )
