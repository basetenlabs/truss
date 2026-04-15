import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import RLock
from typing import List, Optional

import httpx
import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge import AutoBridge
from torch.optim import AdamW
from transformers import AutoTokenizer

from trainers_server.shared.models import (
    Datum,
    ForwardBackwardDetails,
    ForwardBackwardResult,
    OptimStepDetails,
    OptimStepResult,
    SampleDetails,
    SampledSequence,
    SampleResult,
    SaveStateResult,
    ToInferenceResult,
)

from .models import (
    RLControllerConfig,
    StatusResult,
)

logger = logging.getLogger(__name__)


_ROLLOUT_STARTUP_TIMEOUT_SECONDS = 300
_ROLLOUT_RETRY_INTERVAL_SECONDS = 5


class RLController:
    """Controller that exposes discrete RL training/inference transitions."""

    def __init__(
        self,
        config: RLControllerConfig,
        *,
        bridge=None,
        model_list=None,
        tokenizer=None,
    ) -> None:
        init_t0 = time.perf_counter()
        logger.info("RLController.__init__ starting for model=%s", config.model_id)
        self.config = config
        self._lock = RLock()
        self.mode = "training"
        self.step = 0
        self.last_loss: Optional[float] = None
        self._closed = False
        self._rollout_process: Optional[subprocess.Popen] = None
        self._rollout_port = self._find_free_port()
        self._rollout_max_model_len = max(self.config.training.max_length, 4096)

        if not self.config.training.gpus:
            raise ValueError("training.gpus must contain at least one GPU id.")
        if not self.config.inference.gpus:
            raise ValueError("inference.gpus must contain at least one GPU id.")

        training_device = self._training_device()

        if bridge is None or model_list is None:
            logger.info(
                "Initializing torch.distributed for Megatron parallel state (world_size=1) ..."
            )
            if not dist.is_initialized():
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", str(self._find_free_port()))
                dist.init_process_group(backend="nccl", world_size=1, rank=0)

            logger.info(
                "Loading model via AutoBridge for %s (dtype=bfloat16, device=%s) ...",
                config.model_id,
                training_device,
            )
            t0 = time.perf_counter()
            bridge = AutoBridge.from_hf_pretrained(
                config.model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": training_device},
            )
            provider = bridge.to_megatron_provider()
            provider.tensor_model_parallel_size = 1
            provider.finalize()
            model_list = provider.provide_distributed_model(wrap_with_ddp=False)
            logger.info("Model loaded via AutoBridge in %.1fs", time.perf_counter() - t0)

        self.bridge = bridge
        self.model_list = model_list
        self.model = model_list[0]
        self.model.train()

        if tokenizer is None:
            logger.info("Loading tokenizer for %s ...", config.model_id)
            tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.tokenizer = tokenizer

        logger.info("Initializing AdamW optimizer (params set per optim_step via AdamParams) ...")
        self.optimizer = AdamW(self.model.parameters(), lr=0.0, weight_decay=0.0)
        self.optimizer.zero_grad(set_to_none=True)

        logger.info("Launching vLLM rollout server on inference GPUs=%s ...", config.inference.gpus)
        self._launch_rollout_server()
        logger.info("Waiting for vLLM rollout server at 127.0.0.1:%d ...", self._rollout_port)
        self._wait_for_rollout()
        logger.info("RLController.__init__ complete in %.1fs — mode=%s", time.perf_counter() - init_t0, self.mode)

    def _training_device(self) -> str:
        return f"cuda:{self.config.training.gpus[0]}"

    def _launch_rollout_server(self) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.config.inference.gpus)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_id,
            "--host", "127.0.0.1",
            "--port", str(self._rollout_port),
            "--dtype", "bfloat16",
            "--tensor-parallel-size", str(self.config.inference.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.inference.gpu_memory_utilization),
            "--max-model-len", str(self._rollout_max_model_len),
            "--enforce-eager",
        ]
        self._rollout_process = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
        )

    def _wait_for_rollout(self) -> None:
        deadline = time.time() + float(_ROLLOUT_STARTUP_TIMEOUT_SECONDS)
        last_error: Optional[Exception] = None
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            if self._rollout_process is not None and self._rollout_process.poll() is not None:
                raise RuntimeError(
                    f"Rollout server exited during startup (code={self._rollout_process.returncode})."
                )

            logger.info(
                "Waiting for rollout server (attempt %d, port=%d) ...",
                attempt,
                self._rollout_port,
            )
            try:
                resp = requests.get(
                    f"http://127.0.0.1:{self._rollout_port}/health",
                    timeout=float(_ROLLOUT_RETRY_INTERVAL_SECONDS),
                )
                if resp.status_code == 200:
                    logger.info("Rollout server is ready.")
                    return
                raise RuntimeError(f"Health endpoint returned status={resp.status_code}")
            except Exception as e:
                last_error = e
                logger.info("Rollout server not ready yet: %s", e)
                time.sleep(float(_ROLLOUT_RETRY_INTERVAL_SECONDS))

        raise RuntimeError(
            f"Timed out waiting for rollout server after {_ROLLOUT_STARTUP_TIMEOUT_SECONDS}s. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    def _model_device(self) -> torch.device:
        return torch.device(self._training_device())

    def _ensure_training_mode(self, caller: str) -> None:
        if self.mode == "training":
            return
        logger.info("%s: auto-switching from inference to training mode ...", caller)
        self.model.train()
        self.mode = "training"

    @staticmethod
    def _extract_reward(datum: Datum) -> float:
        reward_tensor = datum.loss_fn_inputs.get("reward")
        if reward_tensor is None:
            return 1.0
        value = reward_tensor.data
        while isinstance(value, list):
            if not value:
                raise ValueError("loss_fn_inputs.reward must not be empty.")
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("loss_fn_inputs.reward must be numeric.") from exc

    def forward_backward(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
        if not details.data:
            raise ValueError("`data` must contain at least one request.")
        with self._lock:
            self._ensure_training_mode("forward_backward")
            if details.loss_fn != "cross_entropy":
                raise ValueError(f"Unsupported loss_fn: {details.loss_fn}. Only 'cross_entropy' is supported.")
            t0 = time.perf_counter()
            logger.info("forward_backward: preparing %d pre-tokenized sample(s) ...", len(details.data))
            self.model.train()

            batch_tokens: List[List[int]] = []
            reward_values = []
            for datum in details.data:
                tokens = datum.model_input.to_ints()
                if len(tokens) < 2:
                    raise ValueError("Each datum.model_input must contain at least 2 tokens.")
                batch_tokens.append(tokens)
                reward_values.append(self._extract_reward(datum))

            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id or 0

            max_len = max(len(tokens) for tokens in batch_tokens)
            input_ids_rows: List[List[int]] = []
            attention_rows: List[List[int]] = []
            label_rows: List[List[int]] = []
            for tokens in batch_tokens:
                pad_len = max_len - len(tokens)
                input_ids_rows.append(tokens + [pad_token_id] * pad_len)
                attention_rows.append([1] * len(tokens) + [0] * pad_len)
                label_rows.append(tokens + [-100] * pad_len)

            device = self._model_device()
            input_ids = torch.tensor(input_ids_rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
            labels = torch.tensor(label_rows, dtype=torch.long, device=device)

            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            logger.info("forward_backward: running forward pass ...")
            output_tensor = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # Megatron output: [batch_size * seq_len, vocab_size] — reshape to [B, T, V]
            logits = output_tensor.view(batch_size, seq_len, -1)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = shift_logits.shape[-1]

            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, vocab_size),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.shape)

            # Compute per-token logprobs (log_softmax, then gather target tokens).
            with torch.no_grad():
                log_probs = F.log_softmax(shift_logits, dim=-1)
                gather_labels = shift_labels.clone()
                gather_labels[gather_labels == -100] = 0
                token_logprobs = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)

            valid_mask = shift_labels.ne(-100)
            token_loss = token_loss * valid_mask
            valid_counts = valid_mask.sum(dim=-1).clamp(min=1)
            per_sample_loss = token_loss.sum(dim=-1) / valid_counts

            rewards = torch.tensor(reward_values, dtype=per_sample_loss.dtype, device=per_sample_loss.device)
            loss = (per_sample_loss * rewards).mean()

            logger.info("forward_backward: running backward pass ...")
            loss.backward()
            self.last_loss = float(loss.detach().cpu())

            # Build per-sample logprobs output (masked positions excluded).
            loss_fn_outputs = []
            for i in range(token_logprobs.shape[0]):
                mask_i = valid_mask[i]
                lp = token_logprobs[i][mask_i].detach().cpu().tolist()
                loss_fn_outputs.append({"logprobs": {
                    "data": lp,
                    "dtype": "float32",
                    "shape": [len(lp)],
                }})

            logger.info(
                "forward_backward: done in %.2fs — loss=%.4f, num_samples=%d",
                time.perf_counter() - t0,
                self.last_loss,
                len(details.data),
            )
            return ForwardBackwardResult(
                loss_fn_output_type="per_token_logprobs",
                loss_fn_outputs=loss_fn_outputs,
                metrics={"loss": self.last_loss},
            )

    def optim_step(self, details: OptimStepDetails) -> OptimStepResult:
        with self._lock:
            self._ensure_training_mode("optim_step")
            t0 = time.perf_counter()
            logger.info("optim_step: stepping optimizer (current step=%d) ...", self.step)

            params = details.adam_params
            for pg in self.optimizer.param_groups:
                pg["lr"] = params.learning_rate
                pg["betas"] = (params.beta1, params.beta2)
                pg["eps"] = params.eps
                pg["weight_decay"] = params.weight_decay

            grad_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(p.grad.detach().float().pow(2).sum().item())
            grad_norm = grad_norm_sq ** 0.5

            if params.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), params.grad_clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            logger.info(
                "optim_step: done in %.2fs — step=%d, lr=%.2e, grad_norm=%.4f",
                time.perf_counter() - t0,
                self.step,
                learning_rate,
                grad_norm,
            )
            return OptimStepResult(metrics={
                "step": float(self.step),
                "learning_rate": learning_rate,
                "lr": learning_rate,
                "grad_norm": grad_norm,
            })

    def to_inference(self) -> ToInferenceResult:
        with self._lock:
            t0 = time.perf_counter()
            logger.info("to_inference: switching to inference mode ...")
            self.model.eval()
            self.mode = "inference"
            # Weight sync to rollout server will be wired here once MegatronWeightWriter
            # is available (Will's baseten_weight_sync integration).
            logger.info("to_inference: weight sync skipped (not yet implemented) ...")
            logger.info("to_inference: done in %.2fs — mode=%s", time.perf_counter() - t0, self.mode)
            return ToInferenceResult(mode=self.mode)

    def to_training(self) -> StatusResult:
        with self._lock:
            t0 = time.perf_counter()
            logger.info("to_training: switching from inference to training mode ...")
            self.model.train()
            self.mode = "training"
            logger.info("to_training: done in %.2fs — mode=%s", time.perf_counter() - t0, self.mode)
            return self.get_status()

    def sample(self, details: SampleDetails) -> SampleResult:
        with self._lock:
            if self.mode != "inference":
                raise RuntimeError("sample is only allowed in inference mode.")
            t0 = time.perf_counter()
            prompt_tokens = details.prompt.to_ints()
            params = details.sampling_params
            max_tokens = params.max_tokens or 128
            remaining_budget = self._rollout_max_model_len - len(prompt_tokens)
            effective_max_tokens = max(1, min(max_tokens, remaining_budget))

            logger.info(
                "sample: %d prompt tokens, num_samples=%d, max_tokens=%d ...",
                len(prompt_tokens),
                details.num_samples,
                effective_max_tokens,
            )

            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            sequences: List[SampledSequence] = []

            for _ in range(details.num_samples):
                resp = httpx.post(
                    f"http://127.0.0.1:{self._rollout_port}/v1/completions",
                    json={
                        "model": self.config.model_id,
                        "prompt": prompt_text,
                        "max_tokens": effective_max_tokens,
                        "temperature": params.temperature,
                        "top_p": params.top_p,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                generated_text = data["choices"][0]["text"]
                generated_tokens = self.tokenizer.encode(generated_text, add_special_tokens=False)
                stop_reason = data["choices"][0].get("finish_reason") or "length"
                sequences.append(SampledSequence(
                    tokens=generated_tokens,
                    logprobs=None,
                    stop_reason=stop_reason,
                ))

            logger.info("sample: done in %.2fs — %d sequence(s)", time.perf_counter() - t0, len(sequences))
            return SampleResult(sequences=sequences)

    def save_state(self, path: str) -> SaveStateResult:
        with self._lock:
            self._ensure_training_mode("save_state")
            t0 = time.perf_counter()
            logger.info("save_state: saving to %s ...", path)
            ckpt_dir = Path(path)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            logger.info("save_state: saving model weights via bridge ...")
            self.bridge.save_hf_pretrained(self.model_list, str(ckpt_dir))
            logger.info("save_state: saving tokenizer ...")
            self.tokenizer.save_pretrained(str(ckpt_dir))

            trainer_state = {
                "step": self.step,
                "mode": self.mode,
                "last_loss": self.last_loss,
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.model_dump(),
            }
            logger.info("save_state: saving trainer state (step=%d) ...", self.step)
            torch.save(trainer_state, ckpt_dir / "trainer_state.pt")
            logger.info("save_state: done in %.2fs", time.perf_counter() - t0)
            return SaveStateResult(mode=self.mode)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._rollout_process is not None:
                self._kill_rollout_process_tree()

    def _kill_rollout_process_tree(self) -> None:
        if self._rollout_process is None or self._rollout_process.poll() is not None:
            return
        pid = self._rollout_process.pid
        if pid is None:
            return

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as e:
            logger.warning("Failed to SIGTERM rollout process group: %s", e)
            self._rollout_process.terminate()

        try:
            self._rollout_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Rollout process group did not exit in time, force killing.")
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception as e:
                logger.warning("Failed to SIGKILL rollout process group: %s", e)
                self._rollout_process.kill()
            self._rollout_process.wait(timeout=5)

    def get_status(self) -> StatusResult:
        with self._lock:
            device = self._training_device()
            gpu_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f"cuda:{i}"] = int(torch.cuda.memory_allocated(i))
            grad_norm_sq = 0.0
            has_grad = False
            for param in self.model.parameters():
                if param.grad is None:
                    continue
                has_grad = True
                grad_norm_sq += float(torch.sum(param.grad.detach().float() ** 2).item())
            grad_norm = grad_norm_sq ** 0.5 if has_grad else None
            return StatusResult(
                mode=self.mode,
                step=self.step,
                model_id=self.config.model_id,
                device=device,
                last_loss=self.last_loss,
                grad_norm=grad_norm,
                gpu_memory=gpu_memory,
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
