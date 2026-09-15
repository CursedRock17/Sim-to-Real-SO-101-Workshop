# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone tests for Gate 1 (run: python3 test_gate1.py). No Isaac/torch needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate1 import Gate1Spec, evaluate_trajectory, gate1_fields, gate1_row  # noqa: E402


def _minjerk(T, a, b):
    """5th-order minimum-jerk profile a->b over T samples (maximally smooth)."""
    t = np.linspace(0, 1, T)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    return a + (b - a) * s


def _smooth(T=60):
    tr = np.zeros((T, 6))
    tr[:, 0] = _minjerk(T, 0.0, 50.0)      # one joint sweeps smoothly
    tr[:, 1] = _minjerk(T, 0.0, -20.0)
    return tr


def _jerky(T=60, seed=0):
    rng = np.random.default_rng(seed)
    tr = _smooth(T).copy()
    tr[:, 0] += rng.normal(0, 8.0, T)       # high-freq noise -> big jerk
    tr[:, 1] += rng.normal(0, 8.0, T)
    return tr


def test_smooth_passes_and_beats_jerky():
    s = evaluate_trajectory(_smooth())
    j = evaluate_trajectory(_jerky())
    assert s.verdict == "PASS", s
    assert s.peak_jerk < j.peak_jerk, (s.peak_jerk, j.peak_jerk)
    assert s.ldlj > j.ldlj, (s.ldlj, j.ldlj)                # smoother = higher LDLJ
    assert s.quality_score > j.quality_score, (s.quality_score, j.quality_score)


def test_jerky_trips_flag_with_calibrated_cap():
    s = evaluate_trajectory(_smooth())
    j_metrics = evaluate_trajectory(_jerky())
    # set the cap between the smooth and jerky peak jerks -> only jerky flags
    cap = 0.5 * (s.peak_jerk + j_metrics.peak_jerk)
    spec = Gate1Spec(jerk_cap=cap)
    assert "jerky" not in evaluate_trajectory(_smooth(), spec=spec).flags
    assert "jerky" in evaluate_trajectory(_jerky(), spec=spec).flags


def test_idle_trim_suggested():
    T = 60
    tr = np.zeros((T, 6))
    tr[15:45, 0] = _minjerk(30, 0.0, 40.0)   # static 0..15 and 45..60, active middle
    r = evaluate_trajectory(tr)
    assert 12 <= r.lead_idle <= 16, r.lead_idle
    assert 12 <= r.trail_idle <= 16, r.trail_idle
    assert r.trim_start == r.lead_idle and r.trim_end == T - r.trail_idle, r


def test_excessive_interior_idle_flags():
    # active start/end but frozen in the middle -> interior idle high
    T = 80
    tr = np.zeros((T, 6))
    tr[:10, 0] = _minjerk(10, 0.0, 30.0)
    tr[10:70, 0] = 30.0                       # frozen middle
    tr[70:, 0] = _minjerk(10, 30.0, 60.0)
    r = evaluate_trajectory(tr)
    assert "excessive_idle" in r.flags, r


def test_too_short_flags_and_disqualifies():
    r = evaluate_trajectory(_smooth(T=10))
    assert "too_short" in r.flags and r.verdict == "FLAG", r


def test_duration_outlier_with_ref():
    r = evaluate_trajectory(_smooth(T=120), ref_len=30)     # 4x median
    assert "duration_outlier" in r.flags, r
    ok = evaluate_trajectory(_smooth(T=60), ref_len=55)     # ~1.09x
    assert "duration_outlier" not in ok.flags, ok


def test_progress_monotonic_vs_dithering():
    T = 60
    good = np.linspace(0.3, 0.0, T)                          # steadily closes on goal
    r_good = evaluate_trajectory(_smooth(T), block_to_goal=good)
    assert "low_progress" not in r_good.flags and r_good.progress_frac > 0.9, r_good
    dither = 0.3 - 0.3 * (np.sin(np.linspace(0, 6 * np.pi, T)) * 0.5 + 0.5)  # in-and-out
    r_bad = evaluate_trajectory(_smooth(T), block_to_goal=dither)
    assert "low_progress" in r_bad.flags, r_bad


def test_row_matches_fields():
    r = evaluate_trajectory(_jerky())
    row = gate1_row(r)
    assert set(row.keys()) == set(gate1_fields())
    assert isinstance(row["flags"], str)                    # joined for CSV


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
    print(f"\n{passed}/{len(tests)} gate1 tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
