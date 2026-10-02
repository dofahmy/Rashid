#!/usr/bin/env python3
"""
Self-test for Rajih US SECOND_STOP_RECOVERY system.

Run from the project root on Railway:
    python test_second_stop_system.py

This test is isolated:
- does NOT connect to or modify the production database
- does NOT send Telegram messages
- does NOT create customer publications
- uses in-memory fake sessions and synthetic 15-minute bars

It verifies:
1) Every new US conditional plan is converted to the second-stop system.
2) Level 2 = original stop - 1.2 ATR.
3) Level 3 = level 2 - 1.2 ATR.
4) Target = original first stop.
5) Order type is LIMIT.
6) Price above level 2 does not activate the plan.
7) Touching level 2 activates the plan.
8) After activation, touching level 1 produces TARGET.
9) After activation, touching level 3 produces STOPPED.
10) If both target and stop occur in one OHLC bar, stop wins conservatively.
11) While waiting for level 2, trading above level 1 does NOT mark the plan MISSED.
12) Waiting order expires at 78 bars.
13) A gap at/below level 3 before a safe limit fill is CANCELLED.
14) Legacy/non-second-stop plans still keep the old minimum-RR gate.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

from monitor import engine
from monitor.models import Plan


class FakeSession:
    """Minimal SQLAlchemy-session-like object used only by this test."""
    def __init__(self):
        self.added = []
        self._next_id = 1000

    def scalar(self, statement):
        # new_plan duplicate/open-plan checks and event duplicate checks:
        # isolated test assumes no prior records.
        return None

    def get(self, model, ident):
        # engine.new_plan reads Setting rows with Session.get().
        # The isolated self-test intentionally supplies no DB settings, so
        # returning None makes engine.policy()/new_plan use their normal defaults.
        return None

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None and isinstance(obj, Plan):
            obj.id = self._next_id
            self._next_id += 1

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None and isinstance(obj, Plan):
                obj.id = self._next_id
                self._next_id += 1


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, s, p, kind, ts, details=None):
        self.events.append({
            "plan_id": getattr(p, "id", None),
            "symbol": p.symbol,
            "kind": kind,
            "ts": int(ts),
            "details": details or {},
        })


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def assert_eq(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected={expected!r}, actual={actual!r}")


def assert_approx(actual, expected, msg, tol=1e-6):
    if not approx(actual, expected, tol):
        raise AssertionError(f"{msg}: expected≈{expected!r}, actual={actual!r}")


def make_result():
    # Chosen to make expected levels obvious:
    # original stop = 100
    # ATR = 2
    # second stop = 100 - 1.2*2 = 97.60
    # third stop  = 97.60 - 1.2*2 = 95.20
    # target      = 100
    return {
        "conditional_plan": True,
        "entry_reference": 102.0,
        "stop_reference": 100.0,
        "selected_target_price": 106.0,
        "atr14": 2.0,
        "technical_score_100": 55.0,
        "signal_bar_close": 102.0,
        # audit fields commonly carried by strategy result
        "feed_last_price": 102.0,
    }


def make_stock(market="US"):
    return SimpleNamespace(symbol="TEST", market=market, last_price=102.0)


def make_plan():
    s = FakeSession()
    p = engine.new_plan(
        s,
        make_stock(),
        make_result(),
        ts=1_000_000,
        settings=engine.policy(),
    )
    assert_true(p is not None, "new_plan should create a US second-stop plan")
    return s, p


def bar(ts, o, h, l, c, v=1000):
    return (int(ts), float(o), float(h), float(l), float(c), float(v))


def test_plan_levels():
    _, p = make_plan()
    ctx = json.loads(p.context_json)

    assert_eq(p.state, "WAITING", "new plan state")
    assert_eq(ctx.get("entry_system"), "SECOND_STOP_RECOVERY", "entry system")
    assert_eq(ctx.get("order_type"), "LIMIT", "order type")

    assert_approx(ctx["original_entry_reference"], 102.0, "original entry retained")
    assert_approx(ctx["original_first_stop"], 100.0, "first stop retained")
    assert_approx(ctx["original_selected_target"], 106.0, "original target retained")

    assert_approx(p.entry, 97.60, "level 2 / buy-limit")
    assert_approx(p.stop, 95.20, "level 3 / protective stop")
    assert_approx(p.target, 100.00, "recovery target / level 1")
    assert_true(p.stop < p.entry < p.target, "required price ordering stop < entry < target")

    assert_eq(engine.order_type(p), "LIMIT", "second-stop plan must always be LIMIT")


def test_waiting_above_level2_does_not_activate_or_miss():
    _, p = make_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        changed = engine.advance(
            FakeSession(), p,
            bar(1_000_900, 101.0, 103.0, 99.5, 102.0),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    # High is above target level 1, but the second-stop system must keep waiting.
    assert_true(not changed, "waiting plan should remain unchanged above level 2")
    assert_eq(p.state, "WAITING", "must not become MISSED before level-2 fill")
    assert_true(p.paper_entry is None, "must not fill before level 2 is touched")


def test_touch_level2_activates():
    _, p = make_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        changed = engine.advance(
            FakeSession(), p,
            bar(1_000_900, 99.0, 99.4, 97.50, 98.10),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    assert_true(changed, "touching level 2 should change state")
    assert_eq(p.state, "ACTIVE", "touching level 2 should activate")
    assert_approx(p.paper_entry, 97.60, "paper fill should be level 2")
    assert_true(p.activation_ts == 1_000_900, "activation timestamp")
    assert_true(any(e["kind"] == "ACTIVE" for e in rec.events), "ACTIVE event recorded")


def test_gap_improvement_at_level2():
    _, p = make_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        # Open below level 2 but safely above level 3 -> better fill at open.
        engine.advance(
            FakeSession(), p,
            bar(1_000_900, 97.30, 98.0, 96.90, 97.70),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    assert_eq(p.state, "ACTIVE", "safe gap below level 2 should activate")
    assert_approx(p.paper_entry, 97.30, "resting limit should receive opening price improvement")


def activated_plan():
    _, p = make_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        engine.advance(
            FakeSession(), p,
            bar(1_000_900, 99.0, 99.2, 97.50, 98.0),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event
    assert_eq(p.state, "ACTIVE", "precondition: plan activated")
    return p


def test_recovery_to_first_stop_is_target():
    p = activated_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        changed = engine.advance(
            FakeSession(), p,
            bar(1_001_800, 98.0, 100.10, 97.80, 100.0),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    assert_true(changed, "target touch should change state")
    assert_eq(p.state, "TARGET", "level 1 should be target")
    assert_approx(p.exit_price, 100.0, "target exit price")


def test_third_stop_is_protective_stop():
    p = activated_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        changed = engine.advance(
            FakeSession(), p,
            bar(1_001_800, 97.2, 97.5, 95.10, 95.4),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    assert_true(changed, "third-stop touch should change state")
    assert_eq(p.state, "STOPPED", "level 3 should stop the trade")
    assert_approx(p.exit_price, 95.20, "protective stop exit")


def test_same_bar_target_and_stop_uses_stop_first():
    p = activated_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        engine.advance(
            FakeSession(), p,
            bar(1_001_800, 97.8, 100.50, 95.00, 98.0),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    assert_eq(p.state, "STOPPED", "ambiguous target+stop candle must use conservative stop-first")
    assert_approx(p.exit_price, 95.20, "same-bar conservative exit")


def test_expiry_is_78_bars():
    _, p = make_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        # Bars remain between level 2 and higher prices, so no fill.
        for i in range(1, 79):
            changed = engine.advance(
                FakeSession(), p,
                bar(1_000_000 + 900*i, 101.0, 102.0, 99.0, 101.0),
                ratio=1.0,
                contiguous=True,
            )
            if i < 78:
                assert_eq(p.state, "WAITING", f"must still wait at bar {i}")
        assert_eq(p.state, "EXPIRED", "must expire on waiting bar 78")
        assert_true(changed, "expiry must report a state change")
    finally:
        engine.event = old_event


def test_gap_at_or_below_level3_cancels_before_fill():
    _, p = make_plan()
    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        engine.advance(
            FakeSession(), p,
            bar(1_000_900, 95.10, 96.0, 94.8, 95.5),
            ratio=1.0,
            contiguous=True,
        )
    finally:
        engine.event = old_event

    assert_eq(p.state, "CANCELLED", "unsafe gap through level 3 must cancel instead of pretending safe fill")
    assert_true(p.paper_entry is None, "cancelled gap must not create a paper fill")


def test_second_stop_rr_is_about_one_to_one():
    _, p = make_plan()
    rr = (p.target - p.entry) / (p.entry - p.stop)
    assert_true(0.95 <= rr <= 1.05, f"second-stop recovery setup should be ~1:1, got {rr:.4f}")


def test_legacy_rr_gate_still_works():
    # Hand-built non-second-stop plan with RR below legacy minimum 1.5.
    p = Plan(
        symbol="LEGACY",
        market="US",
        strategy_version=engine.VERSION,
        signal_ts=2_000_000,
        last_bar=2_000_000,
        state="WAITING",
        entry=100.0,
        stop=99.0,
        target=101.0,
        atr=1.0,
        score=50.0,
        waiting_bars=0,
        retest_bars=0,
        context_json=json.dumps({"order_type": "LIMIT"}),
        policy_json=json.dumps(engine.policy(), sort_keys=True),
    )
    p.id = 9999

    rec = Recorder()
    old_event = engine.event
    engine.event = rec
    try:
        ok, reason = engine._activate_fill(FakeSession(), p, 100.0, 2_000_900, "LIMIT")
    finally:
        engine.event = old_event

    assert_true(not ok, "legacy low-RR setup must still be rejected")
    assert_eq(reason, "rr_below_minimum_after_fill", "legacy RR rejection reason")


TESTS = [
    test_plan_levels,
    test_waiting_above_level2_does_not_activate_or_miss,
    test_touch_level2_activates,
    test_gap_improvement_at_level2,
    test_recovery_to_first_stop_is_target,
    test_third_stop_is_protective_stop,
    test_same_bar_target_and_stop_uses_stop_first,
    test_expiry_is_78_bars,
    test_gap_at_or_below_level3_cancels_before_fill,
    test_second_stop_rr_is_about_one_to_one,
    test_legacy_rr_gate_still_works,
]


def main():
    passed = 0
    failed = 0
    print("Rajih SECOND_STOP_RECOVERY self-test")
    print("=" * 48)
    print("Production DB: NOT USED")
    print("Telegram/customer delivery: NOT USED")
    print()

    for fn in TESTS:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"❌ {fn.__name__}")
            print(f"   {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)
        else:
            passed += 1
            print(f"✅ {fn.__name__}")

    print()
    print("=" * 48)
    print(f"Passed: {passed}/{len(TESTS)}")
    print(f"Failed: {failed}/{len(TESTS)}")

    if failed:
        print("RESULT: ❌ SECOND-STOP SYSTEM TEST FAILED")
        return 1

    print("RESULT: ✅ SECOND-STOP SYSTEM LOGIC PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
