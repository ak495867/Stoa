import json

from stoa.cli import run


def test_validate_command(capsys):
    code = run(["validate", "examples/basic_config.json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["assets"] == ["US_EQ", "GOV_BOND", "GOLD"]


def test_optimize_command_writes_output(tmp_path):
    output = tmp_path / "result.json"
    code = run(["optimize", "examples/basic_config.json", "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["weights"]) == {"US_EQ", "GOV_BOND", "GOLD"}
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-7


def test_invalid_file_returns_error(capsys, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")
    code = run(["validate", str(path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "stoa:" in captured.err
