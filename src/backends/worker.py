"""Persistent JSON-lines worker transport; algorithm dependencies stay in the child."""
import json
import subprocess
from pathlib import Path

from .base import Backend, BackendResult


class WorkerBackend(Backend):
    def __init__(self, cfg, project_root, out_dir):
        command = cfg["worker_command"]
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError("worker_command must be a nonempty argv list, not a shell string")
        self.cfg = {k: v for k, v in cfg.items() if k != "worker_command"}
        logs = Path(out_dir) / "logs"
        logs.mkdir(exist_ok=True)
        self.log = (logs / "worker.log").open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [*command, str(project_root / "src/worker.py")], cwd=project_root,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding="utf-8", bufsize=1,
            )
        except BaseException:
            self.log.close()
            raise

    def infer(self, sample, cfg, out_dir):
        if self.process.poll() is not None:
            raise RuntimeError("Worker exited; inspect logs/worker.log")
        self.process.stdin.write(json.dumps({"config": self.cfg, "sample": sample, "out_dir": out_dir}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Worker closed protocol; inspect logs/worker.log")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return BackendResult(**response["result"])

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.log.close()
