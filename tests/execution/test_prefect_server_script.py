import socket
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "start-prefect-server.sh"


def test_prefect_server_script_has_valid_bash_syntax():
    completed = subprocess.run(
        ["bash", "-n", SCRIPT_PATH],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_prefect_server_script_rejects_sourcing():
    completed = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", SCRIPT_PATH],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "Execute this script; do not source it" in completed.stderr


def test_prefect_server_script_supports_help():
    completed = subprocess.run(
        [SCRIPT_PATH, "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0
    assert "--port PORT" in completed.stdout
    assert "--connect-host HOST" in completed.stdout


def test_prefect_server_script_rejects_invalid_port():
    completed = subprocess.run(
        [SCRIPT_PATH, "--port", "invalid"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "--port must be an integer" in completed.stderr


def test_prefect_server_script_rejects_occupied_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        completed = subprocess.run(
            [SCRIPT_PATH, "--host", "127.0.0.1", "--port", str(port)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    assert completed.returncode == 2
    assert f"Port {port} is already in use" in completed.stderr


def test_prefect_server_script_only_starts_supporting_service():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "prefect server start" in script
    assert "PREFECT_SERVER_ANALYTICS_ENABLED=false" in script
    assert "PREFECT_UI_API_URL=/api" in script
    assert "DO_NOT_TRACK=1" in script
    assert 'export PREFECT_API_URL="$API_URL"' in script
    assert "command -v python3" in script
    assert "rapthor rapthor.parset" in script
    assert not any(line.startswith("rapthor ") for line in script.splitlines())
