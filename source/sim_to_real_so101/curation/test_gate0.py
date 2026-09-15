# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone tests for Gate 0 (run: python3 test_gate0.py). No Isaac/torch needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate0 import BoxTaskSpec, ScorecardWriter, block_in_box, evaluate_episode  # noqa: E402

SPEC = BoxTaskSpec()
IN_BOX = (SPEC.box_xy[0], SPEC.box_xy[1], 0.05)      # settled in the box
SETTLED = (0.0, 0.0, 0.0)


def _clean_success(**over):
    kw = dict(plan_ok=True, block_final_pos=IN_BOX, block_final_vel=SETTLED,
              max_lift_m=0.14, pre_grasp_disp_m=0.002, spec=SPEC)
    kw.update(over)
    return evaluate_episode(**kw)


def test_clean_success_keeps():
    r = _clean_success()
    assert r.verdict == "KEEP" and r.drop_reason == "", r
    assert r.plan_ok and r.approach_clean and r.grasp_ok and r.success


def test_plan_failure_drops_first():
    # even with otherwise-perfect signals, a plan failure drops with plan_failed
    r = _clean_success(plan_ok=False)
    assert r.verdict == "DROP" and r.drop_reason == "plan_failed", r


def test_knocked_block_drops():
    r = _clean_success(pre_grasp_disp_m=0.05)   # block shoved during approach
    assert r.verdict == "DROP" and r.drop_reason == "knocked_block", r


def test_grasp_failure_drops():
    r = _clean_success(max_lift_m=0.004)         # never lifted off the surface
    assert r.verdict == "DROP" and r.drop_reason == "grasp_failed", r


def test_place_outside_box_drops():
    r = _clean_success(block_final_pos=(0.05, 0.0, 0.05))   # dropped short of the box
    assert r.verdict == "DROP" and r.drop_reason == "place_failed", r


def test_in_box_but_still_moving_drops():
    # inside the footprint but not settled -> not a success (block bouncing)
    r = _clean_success(block_final_vel=(0.3, 0.0, 0.0))
    assert r.verdict == "DROP" and r.drop_reason == "place_failed", r


def test_funnel_order_reports_earliest_failure():
    # multiple gates fail; drop_reason must be the earliest in the funnel
    r = _clean_success(plan_ok=False, max_lift_m=0.0, block_final_pos=(0.5, 0.5, 0.5))
    assert r.drop_reason == "plan_failed", r
    r = _clean_success(pre_grasp_disp_m=0.09, max_lift_m=0.0)
    assert r.drop_reason == "knocked_block", r


def test_block_in_box_predicate_edges():
    assert block_in_box(IN_BOX, SETTLED, SPEC)
    assert not block_in_box((0.10, 0.20, 0.20), SETTLED, SPEC)     # above the rim
    assert not block_in_box((0.30, 0.20, 0.05), SETTLED, SPEC)     # outside footprint
    assert not block_in_box(IN_BOX, (1.0, 0.0, 0.0), SPEC)         # not settled


def test_scorecard_writer_yield(tmp_path="/tmp/gate0_scorecard_test.csv"):
    with ScorecardWriter(tmp_path) as sc:
        sc.log("ep0", _clean_success())                     # KEEP
        sc.log("ep1", _clean_success(max_lift_m=0.0))       # DROP grasp
        sc.log("ep2", _clean_success())                     # KEEP
    assert sc.n_total == 3 and sc.n_keep == 2
    summary = os.path.join(os.path.dirname(tmp_path),
                           os.path.basename(tmp_path).rsplit(".", 1)[0] + "_summary.json")
    assert os.path.exists(summary)
    with open(tmp_path) as f:
        assert len(f.readlines()) == 4  # header + 3 rows
    os.remove(tmp_path); os.remove(summary)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{passed}/{len(tests)} gate0 tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
