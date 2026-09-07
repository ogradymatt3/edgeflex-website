"""Brokers: PaperBroker (simulated fills, default) and KrakenBroker (real limit orders).

Live trading requires KRAKEN_API_KEY / KRAKEN_API_SECRET in the environment and --live on the
CLI. Nothing here places a real order unless both are true.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from dataclasses import dataclass


@dataclass
class Fill:
    side: str
    units: float
    price: float
    fee: float


class PaperBroker:
    def __init__(self, cash: float, maker_fee: float = 0.0025, taker_fee: float = 0.0040, slippage: float = 0.0005):
        self.cash, self.units = cash, 0.0
        self.maker_fee, self.taker_fee, self.slippage = maker_fee, taker_fee, slippage
        self.fills: list[Fill] = []

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def order(self, side: str, units: float, price: float, post_only: bool = True) -> Fill:
        fee_rate = self.maker_fee if post_only else self.taker_fee
        px = price * (1 + self.slippage) if side == "buy" else price * (1 - self.slippage)
        notional = units * px
        fee = notional * fee_rate
        if side == "buy":
            self.cash -= notional + fee
            self.units += units
        else:
            self.cash += notional - fee
            self.units -= units
        f = Fill(side, units, px, fee)
        self.fills.append(f)
        return f


class KrakenBroker:
    """Minimal signed REST client. Limit orders only, post-only by default (maker fee)."""

    URL = "https://api.kraken.com"

    def __init__(self, pair: str = "XBTUSD"):
        self.key = os.environ["KRAKEN_API_KEY"]
        self.secret = base64.b64decode(os.environ["KRAKEN_API_SECRET"])
        self.pair = pair

    def _private(self, path: str, data: dict) -> dict:
        import requests

        data = {**data, "nonce": int(time.time() * 1000)}
        post = urllib.parse.urlencode(data)
        sha = hashlib.sha256((str(data["nonce"]) + post).encode()).digest()
        mac = hmac.new(self.secret, f"/0/private/{path}".encode() + sha, hashlib.sha512)
        headers = {"API-Key": self.key, "API-Sign": base64.b64encode(mac.digest()).decode()}
        r = requests.post(f"{self.URL}/0/private/{path}", headers=headers, data=data, timeout=20)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise RuntimeError(body["error"])
        return body["result"]

    def balances(self) -> dict:
        return self._private("Balance", {})

    def order(self, side: str, units: float, price: float, post_only: bool = True) -> dict:
        data = {"pair": self.pair, "type": side, "ordertype": "limit", "price": f"{price:.1f}",
                "volume": f"{units:.8f}"}
        if post_only:
            data["oflags"] = "post"
        return self._private("AddOrder", data)

    def cancel_all(self) -> dict:
        return self._private("CancelAll", {})
