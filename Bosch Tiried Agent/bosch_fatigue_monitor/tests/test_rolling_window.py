"""Tests for RollingWindow — the signal processing foundation."""
import time
import pytest
from pipeline.state_aggregator import RollingWindow


class TestRollingWindow:
    def test_empty_window_returns_zero(self):
        w = RollingWindow(60.0)
        assert w.mean() == 0.0
        assert w.variance() == 0.0
        assert w.count() == 0
        assert w.frequency_per_minute() == 0.0

    def test_single_value(self):
        w = RollingWindow(60.0)
        w.push(5.0)
        assert w.mean() == pytest.approx(5.0)
        assert w.count() == 1

    def test_mean_calculation(self):
        w = RollingWindow(60.0)
        for v in [2.0, 4.0, 6.0, 8.0]:
            w.push(v)
        assert w.mean() == pytest.approx(5.0)

    def test_variance_uniform_values(self):
        """Variance should be 0 when all values are identical."""
        w = RollingWindow(60.0)
        for _ in range(5):
            w.push(10.0)
        assert w.variance() == pytest.approx(0.0)

    def test_variance_spread_values(self):
        """Variance should be nonzero for spread values."""
        w = RollingWindow(60.0)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.push(v)
        assert w.variance() > 0.0

    def test_low_variance_signals_monotonous_driving(self):
        """Core fatigue heuristic: low speed variance = monotonous = fatigued."""
        alert_window = RollingWindow(60.0)
        normal_window = RollingWindow(60.0)

        # Fatigued driver: nearly constant speed
        for v in [60.0, 60.1, 59.9, 60.0, 60.2]:
            alert_window.push(v)

        # Alert driver: varied speed
        for v in [55.0, 62.0, 58.0, 65.0, 61.0]:
            normal_window.push(v)

        assert alert_window.variance() < normal_window.variance()

    def test_eviction_removes_old_entries(self):
        """Values older than window_seconds must be evicted."""
        w = RollingWindow(window_seconds=1.0)
        now = time.time()
        w.push(99.0, ts=now - 2.0)   # 2 seconds old → should be evicted
        w.push(1.0,  ts=now)          # fresh → should stay
        assert w.count() == 1
        assert w.latest() == pytest.approx(1.0)

    def test_frequency_per_minute(self):
        """10 events in a 60-second window → 10/min."""
        w = RollingWindow(60.0)
        now = time.time()
        for i in range(10):
            w.push(1.0, ts=now - i)
        assert w.frequency_per_minute() == pytest.approx(10.0 / 1.0)

    def test_latest_returns_most_recent(self):
        w = RollingWindow(60.0)
        for v in [1.0, 5.0, 3.0]:
            w.push(v)
        assert w.latest() == pytest.approx(3.0)

    def test_window_bounds_clamp_to_one(self):
        """min(1.0, ...) guard: PERCLOS must never exceed 1."""
        w = RollingWindow(60.0)
        for v in [0.5, 0.8, 1.2]:  # 1.2 is out-of-range raw value
            w.push(min(1.0, v))
        assert w.latest() == pytest.approx(1.0)
