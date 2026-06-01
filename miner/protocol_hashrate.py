"""Protocol-share hashrate accounting for Pearl pool benchmarks.

This is intentionally separate from local MAC/s accounting. Pool-visible TH/s must be
derived from accepted share difficulty over time, or read back from the pool/miner log,
not from the CUDA harness throughput.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

DEFAULT_RTX3090_SHARE_DIFF = 32768
DEFAULT_STRATUM_SHARE_UNIT = 2**32

_REJECT_RE = re.compile(r"\b(reject(?:ed)?|invalid|denied)\b", re.IGNORECASE)
_STALE_RE = re.compile(r"\b(stale|expired)\b", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"\b(accept(?:ed)?|share\s+accepted|accepted\s+share)\b", re.IGNORECASE)
_DIFF_RE = re.compile(r"(?:\bdiff(?:iculty)?\b|\bd\b)\s*[=: ]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TH_RE = re.compile(
    r"(?:hashrate|speed|rate|raw|reported|share_equiv|pool|gpu)?[^0-9\n]{0,32}"
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:TH/s|Th/s|terahash(?:es)?/s)",
    re.IGNORECASE,
)
_FIELD_TH_RE = re.compile(
    r"\b(?:hashrate_th_s|raw_th_s|share_equiv_th_s|pool_credited_th_s)\b\s*[=:]\s*"
    r"([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


@dataclass
class ShareStats:
    accepted_shares: int = 0
    rejected_shares: int = 0
    stale_shares: int = 0
    accepted_work: float = 0.0
    reported_th_s: list[float] = field(default_factory=list)
    last_share_diff: float | None = None

    def note_line(self, line: str, default_share_diff: float) -> None:
        diff = _extract_share_diff(line) or self.last_share_diff or default_share_diff
        if diff:
            self.last_share_diff = diff

        if _STALE_RE.search(line):
            self.stale_shares += 1
            return
        if _REJECT_RE.search(line):
            self.rejected_shares += 1
            return
        if _ACCEPT_RE.search(line):
            self.accepted_shares += 1
            self.accepted_work += diff

        th_s = _extract_reported_th_s(line)
        if th_s is not None:
            self.reported_th_s.append(th_s)

    def summary(self, elapsed_sec: float, share_unit: float, power_avg_w: float | None = None) -> dict:
        if elapsed_sec <= 0:
            raise ValueError("elapsed_sec must be positive")
        pool_credited_th_s = self.accepted_work * share_unit / elapsed_sec / 1e12
        raw_th_s = statistics.fmean(self.reported_th_s) if self.reported_th_s else None
        efficiency_th_w = None
        if power_avg_w and power_avg_w > 0:
            efficiency_th_w = (pool_credited_th_s or raw_th_s or 0.0) / power_avg_w
        return {
            "elapsed_sec": round(elapsed_sec, 3),
            "raw_th_s": round(raw_th_s, 6) if raw_th_s is not None else None,
            "pool_credited_th_s": round(pool_credited_th_s, 6),
            "accepted_shares": self.accepted_shares,
            "rejected": self.rejected_shares,
            "stale": self.stale_shares,
            "share_diff": self.last_share_diff,
            "share_unit": share_unit,
            "power_avg_w": round(power_avg_w, 3) if power_avg_w is not None else None,
            "efficiency_th_w": round(efficiency_th_w, 6) if efficiency_th_w is not None else None,
            "reported_samples": len(self.reported_th_s),
        }


def _extract_share_diff(line: str) -> float | None:
    match = _DIFF_RE.search(line)
    return float(match.group(1)) if match else None


def _extract_reported_th_s(line: str) -> float | None:
    field = _FIELD_TH_RE.search(line)
    if field:
        return float(field.group(1))
    match = _TH_RE.search(line)
    return float(match.group(1)) if match else None


def parse_lines(lines, default_share_diff: float = DEFAULT_RTX3090_SHARE_DIFF) -> ShareStats:
    stats = ShareStats(last_share_diff=default_share_diff)
    for line in lines:
        stats.note_line(line, default_share_diff)
    return stats
