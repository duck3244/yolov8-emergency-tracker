"""NotificationManager 억제/조용한 시간 로직 단위 테스트."""
from datetime import datetime, timedelta

import pytest

from notification import NotificationManager


def test_suppression_uses_total_seconds_across_day_boundary():
    """`.seconds` 사용 시 1일 이상 차이가 작게 잘려 알림이 잘못 억제되던 버그 확인."""
    nm = NotificationManager()
    nm.set_alert_rules(notification_interval=60)
    nm.last_alert_time['emergency'] = datetime.now() - timedelta(days=2)
    assert nm._should_suppress_alert('emergency', datetime.now()) is False


def test_recent_alert_is_suppressed():
    nm = NotificationManager()
    nm.set_alert_rules(notification_interval=60)
    nm.last_alert_time['emergency'] = datetime.now() - timedelta(seconds=30)
    assert nm._should_suppress_alert('emergency', datetime.now()) is True


def test_alert_type_based_on_count():
    nm = NotificationManager()
    nm.set_alert_rules(overcrowding_threshold=50, warning_threshold=20)
    assert nm._determine_alert_type(10) is None
    assert nm._determine_alert_type(25) == 'warning'
    assert nm._determine_alert_type(60) == 'emergency'


@pytest.mark.parametrize(
    "hour,quiet_hours,expected",
    [
        (23, (22, 6), True),    # 야간 (22~6) 안쪽
        (3, (22, 6), True),     # 새벽 (22~6) 안쪽
        (10, (22, 6), False),   # 주간
        (12, (9, 18), True),    # 9~18 안쪽
        (20, (9, 18), False),   # 9~18 바깥
    ],
)
def test_quiet_time_window(hour, quiet_hours, expected):
    nm = NotificationManager()
    nm.set_alert_rules(quiet_hours=quiet_hours)
    now = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
    assert nm._is_quiet_time(now) is expected
