"""ConfigManager 로드/저장 및 env 오버라이드 테스트."""
import json
import os

import pytest

from config import ConfigManager


def test_save_does_not_persist_password(tmp_path, monkeypatch):
    """sender_password는 디스크에 절대 저장되지 않아야 한다."""
    # 비밀번호는 env로 주입
    monkeypatch.setenv("SENDER_PASSWORD", "super-secret")
    config_path = tmp_path / "config.json"

    cm = ConfigManager(str(config_path))
    assert cm.email.sender_password == "super-secret"

    cm.save_config()
    with open(config_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["email"]["sender_password"] == ""


def test_env_overrides_smtp_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SENDER_EMAIL", "ops@example.com")

    cm = ConfigManager(str(tmp_path / "config.json"))
    assert cm.email.smtp_server == "smtp.example.com"
    assert cm.email.smtp_port == 2525
    assert cm.email.sender_email == "ops@example.com"


def test_default_config_round_trip(tmp_path):
    config_path = tmp_path / "config.json"
    cm1 = ConfigManager(str(config_path))
    cm1.update_alert_config(overcrowding_threshold=42)
    cm1.save_config()

    cm2 = ConfigManager(str(config_path))
    assert cm2.alert.overcrowding_threshold == 42
