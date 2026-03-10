import os
import signal
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChildProc:
    name: str
    proc: subprocess.Popen


def _venv_python(venv_dir: Path) -> Path | None:
    if os.name == "nt":
        candidate = venv_dir / "Scripts" / "python.exe"
    else:
        candidate = venv_dir / "bin" / "python"
    return candidate if candidate.exists() else None


def _backend_python(repo_root: Path) -> str:
    for venv in (repo_root / "backend" / ".venv", repo_root / ".venv", repo_root / "venv"):
        candidate = _venv_python(venv)
        if candidate is not None:
            return str(candidate)
    return sys.executable


def _popen(*, name: str, args: list[str], cwd: str | None = None) -> ChildProc:
    creationflags = 0
    shell = False
    popen_args: list[str] | str = args

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        exe = shutil.which(args[0])
        # On Windows, .cmd/.bat need to be executed via the shell (cmd.exe).
        if exe is None or exe.lower().endswith((".cmd", ".bat")):
            shell = True
            popen_args = subprocess.list2cmdline(args)
    proc = subprocess.Popen(
        popen_args,
        cwd=cwd,
        creationflags=creationflags,
        shell=shell,
    )
    return ChildProc(name=name, proc=proc)


def _stop(child: ChildProc) -> None:
    if child.proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            # Best-effort stop for Windows: terminate the full process tree.
            subprocess.run(
                ["taskkill", "/PID", str(child.proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        else:
            child.proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    try:
        child.proc.wait(timeout=5)
        return
    except Exception:
        pass

    try:
        child.proc.terminate()
    except Exception:
        return

    try:
        child.proc.wait(timeout=5)
    except Exception:
        try:
            child.proc.kill()
        except Exception:
            pass


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    backend_cwd = repo_root / "backend"
    if not backend_cwd.exists():
        raise SystemExit(f"Missing backend directory: {backend_cwd}")

    frontend_cwd = repo_root / "frontend"
    if not frontend_cwd.exists():
        raise SystemExit(f"Missing frontend directory: {frontend_cwd}")

    python_exe = _backend_python(repo_root)
    backend_args = [
        python_exe,
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
        "--reload-dir",
        str(backend_cwd),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    frontend_args = ["npm", "run", "dev"]

    children: list[ChildProc] = []
    try:
        children.append(_popen(name="backend", args=backend_args, cwd=str(backend_cwd)))
        children.append(_popen(name="frontend", args=frontend_args, cwd=str(frontend_cwd)))

        while True:
            for child in children:
                code = child.proc.poll()
                if code is None:
                    continue
                for other in children:
                    if other is not child:
                        _stop(other)
                return int(code)
    except KeyboardInterrupt:
        for child in children:
            _stop(child)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
