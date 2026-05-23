from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


class BenchmarkRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo_root = Path(config["repo_root"]).resolve()
        self.model_path = self._resolve_path(config["model_path"])
        self.output_dir = Path(config["output_dir"]).resolve()
        self.log_dir = self.output_dir / "logs"
        self.csv_file = self.output_dir / "benchmark_summary.csv"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_report_file = self.output_dir / f"benchmark_report_{timestamp}.json"

        self.server_port = int(config.get("server_port", 8901))
        self.health_url = f"http://127.0.0.1:{self.server_port}/health"
        self.generate_url = f"http://127.0.0.1:{self.server_port}/generate"
        self.alpha = float(config.get("alpha", 0.0))
        self.tokenizer = None
        self.pre_encoded_token_ids: list[int] | None = None
        self.max_prompt_len = int(config.get("max_len", 262144))

        self.report: dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": self._json_safe_config(config),
            },
            "results": [],
        }

        self._ensure_dirs()
        self._init_pre_encoded_prompt()

    def _resolve_path(self, path: str | os.PathLike[str]) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.repo_root / candidate).resolve()

    def _json_safe_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in config.items()}

    def _ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_file.exists():
            with self.csv_file.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "length", "success", "latency_ms",
                    "per_step_latency_ms", "max_memory_mib", "timestamp",
                    "stage"
                ])

    def _init_pre_encoded_prompt(self) -> None:
        self._load_tokenizer()
        if self.tokenizer is None:
            return
        base_prompt = self.config.get("test_prompt", "What is Diffusion-LM?")
        repeat_count = self.max_prompt_len // 7 + 10
        long_prompt = (base_prompt + " ") * repeat_count
        self.pre_encoded_token_ids = self.tokenizer.encode(
            long_prompt, add_special_tokens=False)

    def _load_tokenizer(self) -> None:
        if self.tokenizer is not None or AutoTokenizer is None:
            return
        if not self.model_path.exists():
            print(f"Tokenizer path does not exist: {self.model_path}")
            return
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path),
                                                       trust_remote_code=True)
        if getattr(self.tokenizer, "unk_token", None) is None:
            self.tokenizer.add_special_tokens({"unk_token": "<unk>"})

    def _generate_prompt(self, target_len: int) -> str:
        base_prompt = self.config.get("test_prompt", "What is Diffusion-LM?")
        if self.pre_encoded_token_ids is not None and self.tokenizer is not None:
            if target_len > len(self.pre_encoded_token_ids):
                repeat_count = target_len // 7 + 10
                token_ids = self.tokenizer.encode((base_prompt + " ") * repeat_count,
                                                  add_special_tokens=False)
            else:
                token_ids = self.pre_encoded_token_ids
            return self.tokenizer.decode(token_ids[:target_len])

        repeat_count = max(1, (target_len + 6) // 7)
        return (base_prompt + " ") * repeat_count

    def _save_report(self) -> None:
        with self.json_report_file.open("w", encoding="utf-8") as handle:
            json.dump(self.report, handle, indent=2, ensure_ascii=False)

    def _log_result(self, length: int, success: bool, latency_ms: float,
                    per_step_ms: float, max_memory_mib: int, stage: str) -> None:
        with self.csv_file.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                length, success, latency_ms, per_step_ms, max_memory_mib,
                datetime.now().isoformat(), stage
            ])

    def _gpu_memory_mib(self) -> int:
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits"
            ], encoding="utf-8")
            gpu_ids = self.config.get("gpu_ids")
            selected = set(gpu_ids) if gpu_ids is not None else None
            values = []
            for line in output.strip().splitlines():
                if not line:
                    continue
                index_text, used_text = [part.strip() for part in line.split(",", 1)]
                index = int(index_text)
                if selected is None or index in selected:
                    values.append(int(used_text))
            return max(values) if values else 0
        except Exception:
            return 0

    def _wait_for_server(self, process: subprocess.Popen[str],
                         log_file: Path) -> bool:
        deadline = time.time() + int(self.config.get("server_startup_timeout", 1800))
        print("Waiting for server startup", end="", flush=True)
        while time.time() < deadline:
            if process.poll() is not None:
                print(f"\nServer exited with code {process.returncode}")
                self._print_log_tail(log_file)
                return False
            try:
                response = requests.get(self.health_url, timeout=1)
                if response.status_code == 200:
                    print(" ready")
                    return True
            except requests.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(2)
        print("\nServer startup timeout")
        self._print_log_tail(log_file)
        return False

    def _print_log_tail(self, log_file: Path, lines: int = 40) -> None:
        if not log_file.exists():
            return
        print(f"Last {lines} lines from {log_file}:")
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines()[-lines:]:
            print(line)

    def _request_payload(self, length: int) -> tuple[dict[str, Any], list[int]]:
        alpha = min(max(self.alpha, 0.0), 0.99)
        prompt_len = max(1, int(length * alpha))
        output_len = max(1, length - prompt_len)
        prompt_text = self._generate_prompt(prompt_len)

        target_steps = max(1, int(self.config.get("test_steps", 10)))
        if output_len <= target_steps:
            schedule = [1] * output_len
        else:
            base = output_len // target_steps
            remainder = output_len % target_steps
            schedule = [base + (1 if index < remainder else 0)
                        for index in range(target_steps)]

        payload = {
            "prompt": prompt_text,
            "max_tokens": output_len,
            "temperature": float(self.config.get("test_temperature", 0.0)),
            "gen_length": output_len,
            "output_length": output_len,
            "steps": len(schedule),
            "token_step_schedule": schedule,
            "block_length": int(self.config.get("block_length", output_len)),
            "remasking": self.config.get("test_remasking", "low_confidence"),
            "mask_id": int(self.config["test_mask_id"]),
        }
        return payload, schedule

    def _run_client_test(self, length: int) -> tuple[bool, float, int, list[int]]:
        payload, schedule = self._request_payload(length)
        max_memory = [0]
        stop_monitoring = threading.Event()

        def monitor() -> None:
            while not stop_monitoring.is_set():
                max_memory[0] = max(max_memory[0], self._gpu_memory_mib())
                time.sleep(0.1)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            started = time.perf_counter()
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=float(self.config.get("request_timeout", 64800)),
            )
            response.raise_for_status()
            latency_ms = (time.perf_counter() - started) * 1000
            return True, latency_ms, max_memory[0], schedule
        except Exception as exc:
            print(f"Request failed: {exc}")
            return False, 0.0, max_memory[0], schedule
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=2)

    def _parse_log_file(self, log_file: Path, schedule: list[int]) -> tuple[float, dict[str, Any]]:
        details = {"raw_times": [], "valid_times": [], "schedule_used": schedule}
        if not log_file.exists():
            details["log_status"] = "missing"
            return 0.0, details

        content = log_file.read_text(encoding="utf-8", errors="ignore")
        times = [float(value) for value in re.findall(
            r"execute_diffusion_once time = ([\d.]+) ms", content)]
        details["raw_times"] = times
        details["log_status"] = "parsed"
        if not times:
            return 0.0, details

        warmup_steps = int(self.config.get("warmup_steps_to_drop", 10))
        valid_times = times[warmup_steps:] if len(times) > warmup_steps else times
        details["valid_times"] = valid_times
        return sum(valid_times) / len(valid_times), details

    def _server_env(self) -> dict[str, str]:
        env = os.environ.copy()
        conda_env = self.config.get("conda_env")
        if conda_env:
            conda_env_path = Path(str(conda_env)).expanduser()
            if not conda_env_path.is_absolute():
                conda_prefix = os.environ.get("CONDA_PREFIX")
                if conda_prefix:
                    conda_env_path = Path(conda_prefix).parent / str(conda_env)
            env["PATH"] = f"{conda_env_path / 'bin'}:{env['PATH']}"

        gpu_ids = self.config.get("gpu_ids")
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        env.setdefault("VLLM_USE_CHUNKWISE_GRAPH", "1")
        env.setdefault("VLLM_USAGE_COLLECTION_DISABLED", "1")
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        env.setdefault("VLLM_DO_NOT_TRACK", "1")
        env.setdefault("DO_NOT_TRACK", "1")
        env.setdefault("VLLM_USE_VMM", "1")
        env.setdefault("VLLM_VMM_CHUNK_SIZE", str(2 * 1024 * 1024))
        env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        env.setdefault("MOSAIC_MODEL_ROOT", str(self.model_path.parent))

        model_env_var = self.config.get("model_env_var")
        if model_env_var:
            env[str(model_env_var)] = str(self.model_path)

        for key, value in self.config.get("extra_env", {}).items():
            env[str(key)] = str(value)

        pythonpath = env.get("PYTHONPATH", "")
        entries = [
            str(self.repo_root),
            str(self.repo_root / "flash_sample"),
            str(self.repo_root / "vmm_allocator"),
            str(self.repo_root / "vllm_add_llada" / "cuda_kernels"),
            str(self.model_path),
        ]
        if pythonpath:
            entries.append(pythonpath)
        env["PYTHONPATH"] = ":".join(entries)
        return env

    def _server_command(self, length: int) -> list[str]:
        max_model_len = int(self.config.get("vdllm_max_model_len")
                            or max(8192, length + 512))
        command = [
            str(self.config.get("python_executable", sys.executable)),
            "-m", "vllm.entrypoints.api_server",
            "--model", str(self.model_path),
            "--trust-remote-code",
            "--dtype", str(self.config.get("dtype", "bfloat16")),
            "--host", "127.0.0.1",
            "--port", str(self.server_port),
            "--max-model-len", str(max_model_len),
        ]
        gpu_memory_utilization = self.config.get("gpu_memory_utilization")
        if gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization",
                            str(gpu_memory_utilization)])
        command.extend(self.config.get("extra_vllm_args", []))
        return command

    def run_test_cycle(self, length: int, stage: str = "linear") -> tuple[bool, float, float, int]:
        print(f"\n[{stage}] length={length}")
        server_log = self.log_dir / f"server_{stage}_{length}.log"
        command = self._server_command(length)
        print("Launch:", " ".join(command))

        with server_log.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                env=self._server_env(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True,
            )

        success = False
        latency_ms = 0.0
        max_memory = 0
        schedule: list[int] = []
        try:
            if self._wait_for_server(process, server_log):
                success, latency_ms, max_memory, schedule = self._run_client_test(length)
        finally:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=int(self.config.get("server_shutdown_timeout", 180)))
            except Exception:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    pass

        per_step_ms = 0.0
        details: dict[str, Any] = {}
        if success:
            per_step_ms, details = self._parse_log_file(server_log, schedule)
            if per_step_ms == 0.0 and schedule:
                per_step_ms = latency_ms / len(schedule)

        self._log_result(length, success, latency_ms, per_step_ms, max_memory, stage)
        self.report["results"].append({
            "length": length,
            "stage": stage,
            "success": success,
            "total_latency_ms": latency_ms,
            "per_step_latency_ms": per_step_ms,
            "max_memory_mib": max_memory,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        })
        self._save_report()
        return success, latency_ms, per_step_ms, max_memory

    def run_smart_benchmark(self, start_len: int, step_len: int, max_len: int,
                            precision: int) -> int:
        last_success_len = 0
        first_fail_len = 0
        current_len = start_len

        while current_len <= max_len:
            success, _, _, _ = self.run_test_cycle(current_len, stage="linear")
            if success:
                last_success_len = current_len
                current_len += step_len
            else:
                first_fail_len = current_len
                break

        if first_fail_len == 0:
            return last_success_len
        if last_success_len == 0:
            return 0

        low = last_success_len
        high = first_fail_len
        best_len = last_success_len
        while high - low > precision:
            mid = (low + high) // 2
            if precision > 1:
                mid = (mid // precision) * precision
            if mid <= low:
                mid = low + precision
            if mid >= high:
                break
            success, _, _, _ = self.run_test_cycle(mid, stage="binary")
            if success:
                best_len = mid
                low = mid
            else:
                high = mid
        return best_len