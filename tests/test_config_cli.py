import json

import pytest

from crowdlens.cli import main
from crowdlens.config import Settings


def test_defaults_are_valid():
    assert Settings.load(None).weapon_model is None


def test_invalid_similarity_rejected():
    with pytest.raises(ValueError, match="reid_similarity"):
        Settings(reid_similarity=1.1).validate()


def test_unknown_keys_rejected(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("random_unexpected_field: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown settings"):
        Settings.load(config_path)


def test_invalid_polygon_rejected():
    with pytest.raises(ValueError, match="normalized"):
        Settings(zone_names={"zone": [[0, 0], [4, 0], [1, 1]]}).validate()


def test_cli_check_config(capsys):
    assert main(["check-config"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["person_model"] == "yolov8n.pt"

