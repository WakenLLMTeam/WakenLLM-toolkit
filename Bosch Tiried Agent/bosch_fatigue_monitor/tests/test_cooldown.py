"""Tests for CooldownTracker — spam prevention and adaptive escalation."""
import time
import pytest
from unittest.mock import patch
from orchestrator.cooldown_tracker import CooldownTracker
from models.judge_verdict import SeverityTier
from config import CooldownConfig


@pytest.fixture
def tracker():
    cfg = CooldownConfig(mild_seconds=30, moderate_seconds=60, severe_seconds=120)
    return CooldownTracker(cfg)


class TestCooldownBasic:
    def test_first_alert_always_allowed(self, tracker):
        assert tracker.is_allowed(SeverityTier.MILD) is True
        assert tracker.is_allowed(SeverityTier.MODERATE) is True
        assert tracker.is_allowed(SeverityTier.SEVERE) is True

    def test_none_tier_never_allowed(self, tracker):
        assert tracker.is_allowed(SeverityTier.NONE) is False

    def test_blocked_immediately_after_firing(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.4)
        assert tracker.is_allowed(SeverityTier.MILD) is False

    def test_allowed_after_cooldown_expires(self, tracker):
        with patch("orchestrator.cooldown_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker.record_fired(SeverityTier.MILD, score=0.4)

            mock_time.time.return_value = 1029.0
            assert tracker.is_allowed(SeverityTier.MILD) is False

            mock_time.time.return_value = 1031.0
            assert tracker.is_allowed(SeverityTier.MILD) is True

    def test_tiers_are_independent(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.4)
        assert tracker.is_allowed(SeverityTier.MODERATE) is True
        assert tracker.is_allowed(SeverityTier.SEVERE) is True

    def test_moderate_cooldown_longer_than_mild(self, tracker):
        with patch("orchestrator.cooldown_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker.record_fired(SeverityTier.MILD, score=0.4)
            tracker.record_fired(SeverityTier.MODERATE, score=0.6)

            mock_time.time.return_value = 1035.0
            assert tracker.is_allowed(SeverityTier.MILD) is True
            assert tracker.is_allowed(SeverityTier.MODERATE) is False

    def test_severe_longest_cooldown(self, tracker):
        with patch("orchestrator.cooldown_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker.record_fired(SeverityTier.SEVERE, score=0.8)

            mock_time.time.return_value = 1119.0
            assert tracker.is_allowed(SeverityTier.SEVERE) is False

            mock_time.time.return_value = 1121.0
            assert tracker.is_allowed(SeverityTier.SEVERE) is True


class TestAdaptiveCooldown:
    """Driver ignores alerts → cooldown shortens, tier escalates."""

    def test_ignored_count_increments_when_score_does_not_improve(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.42)  # same level
        assert tracker.ignored_counts[SeverityTier.MILD] == 1

    def test_ignored_count_resets_when_score_drops_significantly(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.50)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.42)  # still high
        tracker.update_ignored(SeverityTier.MILD, current_score=0.30)  # improved (<80%)
        assert tracker.ignored_counts[SeverityTier.MILD] == 0

    def test_cooldown_shortens_after_one_ignore(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.40)  # ignored

        with patch("orchestrator.cooldown_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker.record_fired(SeverityTier.MILD, score=0.40)

            # Base cooldown 30s → 80% = 24s
            mock_time.time.return_value = 1023.0
            assert tracker.is_allowed(SeverityTier.MILD) is False

            mock_time.time.return_value = 1025.0
            assert tracker.is_allowed(SeverityTier.MILD) is True

    def test_cooldown_shortens_further_after_two_ignores(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.41)  # 2nd ignore

        with patch("orchestrator.cooldown_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker.record_fired(SeverityTier.MILD, score=0.40)

            # Base 30s → 60% = 18s
            mock_time.time.return_value = 1017.0
            assert tracker.is_allowed(SeverityTier.MILD) is False

            mock_time.time.return_value = 1019.0
            assert tracker.is_allowed(SeverityTier.MILD) is True

    def test_cooldown_at_minimum_after_three_ignores(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.40)
        for _ in range(3):
            tracker.update_ignored(SeverityTier.MILD, current_score=0.40)

        with patch("orchestrator.cooldown_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker.record_fired(SeverityTier.MILD, score=0.40)

            # Base 30s → 40% = 12s
            mock_time.time.return_value = 1011.0
            assert tracker.is_allowed(SeverityTier.MILD) is False

            mock_time.time.return_value = 1013.0
            assert tracker.is_allowed(SeverityTier.MILD) is True

    def test_tier_escalates_after_two_ignores(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.42)

        assert tracker.effective_tier(SeverityTier.MILD) == SeverityTier.MODERATE

    def test_severe_does_not_escalate_beyond_severe(self, tracker):
        tracker.record_fired(SeverityTier.SEVERE, score=0.80)
        for _ in range(3):
            tracker.update_ignored(SeverityTier.SEVERE, current_score=0.82)

        assert tracker.effective_tier(SeverityTier.SEVERE) == SeverityTier.SEVERE

    def test_no_escalation_before_two_ignores(self, tracker):
        tracker.record_fired(SeverityTier.MILD, score=0.40)
        tracker.update_ignored(SeverityTier.MILD, current_score=0.40)  # only 1 ignore

        assert tracker.effective_tier(SeverityTier.MILD) == SeverityTier.MILD

    def test_no_update_on_first_alert(self, tracker):
        """First alert has no baseline — update_ignored should be a no-op."""
        tracker.update_ignored(SeverityTier.MILD, current_score=0.50)
        assert tracker.ignored_counts[SeverityTier.MILD] == 0
