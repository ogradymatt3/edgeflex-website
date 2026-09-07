"""Account-level guardrails. These exist so a bug or a bad week cannot zero the account."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Limits:
    daily_loss_pct: float = 0.05     # stop trading for the day after -5%
    max_drawdown_pct: float = 0.25   # kill switch: flatten and halt at -25% from peak
    max_position_pct: float = 1.0    # spot: never more than 100% of equity in one asset


class RiskManager:
    def __init__(self, limits: Limits, start_equity: float):
        self.l = limits
        self.peak = start_equity
        self.day_start = start_equity
        self.day = None
        self.halted = False

    def check(self, equity: float, now) -> tuple[bool, str]:
        """Return (allowed_to_trade, reason)."""
        if self.day != now.date():
            self.day, self.day_start = now.date(), equity
        self.peak = max(self.peak, equity)
        if equity <= self.peak * (1 - self.l.max_drawdown_pct):
            self.halted = True
            return False, f"KILL SWITCH: drawdown {equity / self.peak - 1:.1%} from peak"
        if self.halted:
            return False, "halted"
        if equity <= self.day_start * (1 - self.l.daily_loss_pct):
            return False, f"daily loss limit hit ({equity / self.day_start - 1:.1%})"
        return True, "ok"

    def cap_weight(self, w: float) -> float:
        return max(0.0, min(w, self.l.max_position_pct))
