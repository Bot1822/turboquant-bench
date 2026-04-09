from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_markdown_row(path: str | Path, row: list[str]) -> None:
    escaped = [cell.replace("\n", "<br>") for cell in row]
    with Path(path).open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(escaped) + " |\n")


def run_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def completed_to_dict(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
