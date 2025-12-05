# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for pipeline modules."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNotify:
    """Tests for notification module."""

    def test_send_returns_false_when_no_webhook(self, monkeypatch):
        """Test that send returns False when no webhook configured."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("WEBHOOK_URL", raising=False)
        
        from pipelines.notify import send
        
        result = send("Test", "Message", {})
        
        assert result is False

    def test_send_uses_discord_webhook(self, monkeypatch):
        """Test that Discord webhook is used correctly."""
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test")
        
        with patch("pipelines.notify.requests.post") as mock_post:
            mock_post.return_value.ok = True
            
            from pipelines.notify import send
            
            result = send("Test Title", "Test Message", {"key": "value"})
            
            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "discord" in call_args[0][0]

    def test_send_handles_request_error(self, monkeypatch):
        """Test graceful handling of request errors."""
        monkeypatch.setenv("WEBHOOK_URL", "https://example.com/webhook")
        
        with patch("pipelines.notify.requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            from pipelines.notify import send
            
            result = send("Test", "Message", {})
            
            assert result is False


class TestTraffic:
    """Tests for traffic management module."""

    def test_set_canary_makes_request(self, monkeypatch):
        """Test that set_canary makes API request."""
        monkeypatch.setenv("API_URL", "http://test-api:8000")
        
        with patch("pipelines.traffic.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"canary_percent": 25}
            mock_post.return_value = mock_response
            
            from pipelines.traffic import set_canary
            
            result = set_canary(25)
            
            assert result is True
            mock_post.assert_called_once()


class TestPromoteCanary:
    """Tests for canary promotion module."""

    def test_get_versions_empty_dir(self, temp_dir):
        """Test version detection on empty directory."""
        from pipelines.promote_canary import _get_versions
        
        result = _get_versions(temp_dir)
        
        assert result == []

    def test_get_versions_with_versions(self, temp_dir):
        """Test version detection with version directories."""
        (temp_dir / "1").mkdir()
        (temp_dir / "2").mkdir()
        (temp_dir / "3").mkdir()
        (temp_dir / "not_a_version").mkdir()  # Should be ignored
        
        from pipelines.promote_canary import _get_versions
        
        result = _get_versions(temp_dir)
        
        assert sorted(result) == [1, 2, 3]


class TestAutoPromote:
    """Tests for auto-promotion module."""

    def test_query_prometheus_handles_error(self, monkeypatch):
        """Test graceful handling of Prometheus errors."""
        monkeypatch.setenv("PROM_URL", "http://prometheus:9090")
        
        with patch("pipelines.auto_promote.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")
            
            from pipelines.auto_promote import query_prometheus
            
            result = query_prometheus("up")
            
            assert result is None

    def test_query_prometheus_parses_result(self, monkeypatch):
        """Test parsing of Prometheus response."""
        monkeypatch.setenv("PROM_URL", "http://prometheus:9090")
        
        with patch("pipelines.auto_promote.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": {
                    "result": [{"value": [1234567890, "0.75"]}]
                }
            }
            mock_get.return_value = mock_response
            
            from pipelines.auto_promote import query_prometheus
            
            result = query_prometheus("test_metric")
            
            assert result == 0.75
