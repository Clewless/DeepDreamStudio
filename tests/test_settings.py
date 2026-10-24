"""Settings persistence tests."""

import json
from pathlib import Path

from app import settings as settings_module


def _settings_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / settings_module.SETTINGS_FILENAME
    monkeypatch.setattr(settings_module, "settings_path", lambda: path)
    return path


def test_load_settings_returns_defaults_when_missing(tmp_path, monkeypatch) -> None:
    _settings_file(tmp_path, monkeypatch)

    loaded = settings_module.load_settings()

    assert loaded["mode"] == "Single Image"
    assert loaded["guide_path"] == ""
    assert loaded["single_image_iters"] is None


def test_save_and_load_round_trip(tmp_path, monkeypatch) -> None:
    path = _settings_file(tmp_path, monkeypatch)
    payload = dict(settings_module.DEFAULTS)
    payload.update(
        {
            "mode": "Dream Video",
            "single_image_iters": 42,
            "jitter": 0,
            "guide_path": "C:/images/guide.png",
            "target_class": "pizza",
        }
    )

    assert settings_module.save_settings(payload)
    loaded = settings_module.load_settings()

    assert loaded["mode"] == "Dream Video"
    assert loaded["single_image_iters"] == 42
    assert loaded["jitter"] == 0
    assert loaded["guide_path"] == "C:/images/guide.png"
    assert loaded["target_class"] == "pizza"
    assert "unknown" not in json.loads(path.read_text(encoding="utf-8"))


def test_load_settings_survives_corrupt_file(tmp_path, monkeypatch) -> None:
    path = _settings_file(tmp_path, monkeypatch)
    path.write_text("{not json", encoding="utf-8")

    loaded = settings_module.load_settings()

    assert loaded == settings_module.DEFAULTS
