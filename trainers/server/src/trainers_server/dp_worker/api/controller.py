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
import torch.nn.functional as F
from torch.optim import AdamW

from trainers_server.shared.models import (
    ForwardBackwardDetails,
    ForwardBackwardResult,
    OptimStepDetails,
    OptimStepResult,
    SampleDetails,
    SampledSequence,
    SampleResult,
    SaveStateResult,
    TensorData,
    ToInferenceResult,
)

from .models import RLControllerConfig, StatusResult

logger = logging.getLogger(__name__)


_ROLLOUT_STARTUP_TIMEOUT_SECONDS = 300
_ROLLOUT_RETRY_INTERVAL_SECONDS = 5


def _init_ms_swift_megatron() -> None:
    """Set MEGATRON_LM_PATH before importing swift.megatron.

    ms-swift's init_megatron_env() checks this env var; if unset it tries to
    clone megatron-lm from GitHub.  We point it at wherever megatron-core is
    already installed (site-packages on the cluster, the vendor venv locally).
    """
    if "MEGATRON_LM_PATH" not in os.environ:
        import importlib.util
        spec = importlib.util.find_spec("megatron")
        if spec and spec.origin:
            # spec.origin = .../site-packages/megatron/__init__.py
            # parent.parent  = .../site-packages  ← the "repo root" ms-swift expects
            os.environ["MEGATRON_LM_PATH"] = str(Path(spec.origin).parent.parent)
            logger.info("MEGATRON_LM_PATH auto-detected: %s", os.environ["MEGATRON_LM_PATH"])
        else:
            logger.warning("megatron package not found; MEGATRON_LM_PATH not set")
    os.environ.setdefault("USE_HF", "1")


class RLController:
    """Controller that exposes discrete RL training/inference transitions."""

    def __init__(
        self,
        config: RLControllerConfig,
        *,
        model=None,
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

        if model is None or tokenizer is None:
            _init_ms_swift_megatron()

            # Late import: swift.megatron calls init_megatron_env() at import time.
            from megatron.training.initialize import initialize_megatron
            from swift.megatron.arguments import MegatronArguments
            from swift.megatron.model import get_megatron_model_meta
            from swift.megatron.utils import convert_hf_config
            from swift.model import get_model_processor

            logger.info("Loading tokenizer and model config for %s ...", config.model_id)
            _, processor = get_model_processor(
                config.model_id,
                load_model=False,
                use_hf=True,
            )
            tok = getattr(processor, "tokenizer", processor)

            hf_config = processor.model_info.config
            mg_config_kwargs = convert_hf_config(hf_config)
            model_type = mg_config_kwargs.get("hf_model_type", getattr(hf_config, "model_type", None))
            megatron_model_meta = get_megatron_model_meta(model_type)
            if megatron_model_meta is None:
                raise ValueError(
                    f"ms-swift Megatron backend has no registered handler for "
                    f"model_type={model_type!r}. Supported types: qwen2, qwen3, llama, ..."
                )

            logger.info(
                "Building MegatronArguments (model_type=%s, transformer_impl=local, TP=1) ...",
                model_type,
            )
            megatron_args = MegatronArguments(
                model=config.model_id,
                use_hf=True,
                **mg_config_kwargs,
                transformer_impl="local",
                attention_backend="unfused",
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                micro_batch_size=1,
                global_batch_size=1,
                seq_length=config.training.max_length,
                no_save_optim=True,
                no_load_optim=True,
                finetune=True,
            )

            # Point Megatron at the training GPU(s) before init.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(g) for g in config.training.gpus
            )
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", str(self._find_free_port()))
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")

            logger.info("Calling initialize_megatron ...")
            extra_args = megatron_args.parse_to_megatron()
            initialize_megatron(
                extra_args_provider=megatron_model_meta.extra_args_provider,
                args_defaults=extra_args,
            )

            logger.info("Creating Megatron model via model_provider ...")
            t0 = time.perf_counter()
            mg_model = megatron_model_meta.model_provider()
            bridge = megatron_model_meta.bridge_cls()
            logger.info("Loading HF weights into Megatron model from %s ...", config.model_id)
            bridge.load_weights(mg_model, config.model_id)
            logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

            mg_model.cuda()
            mg_model.train()
            model = mg_model
            self.bridge = bridge
            tokenizer = tok

        self.model = model
        self.tokenizer = tokenizer

        logger.info("Initializing AdamW optimizer ...")
        self.optimizer = AdamW(self.model.parameters(), lr=0.0, weight_decay=0.0)
        self.optimizer.zero_grad(set_to_none=True)

        logger.info(
            "Launching vLLM rollout server on inference GPUs=%s ...", config.inference.gpus
        )
        self._launch_rollout_server()
        logger.info("Waiting for vLLM rollout server at 127.0.0.1:%d ...", self._rollout_port)
        self._wait_for_rollout()
        logger.info(
            "RLController.__init__ complete in %.1fs — mode=%s",
            time.perf_counter() - init_t0,
            self.mode,
        )

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

    # ── Loss implementations ─────────────────────────────────────────────

    def _importance_sampling_loss(
        self,
        logits: torch.Tensor,
        target_tokens_list: List[List[int]],
        old_logprobs_list: List[List[float]],
        advantages_list: List[List[float]],
        max_len: int,
        device: torch.device,
        clip_eps: float = 0.2,
    ):
        """PPO-clipped importance sampling loss.

        The datum layout (from the training loop) is:
          model_input  = prompt_tokens + completion_tokens[:-1]
          target_tokens = [0]*ob_len + completion_tokens   (same length as model_input)
          old_logprobs  = [0.0]*ob_len + vllm_logprobs     (same length)
          advantages    = [0.0]*ob_len + [adv]*comp_len    (same length)

        Because model_input already excludes the last completion token, logits[i]
        directly predicts target_tokens[i] with no shift — valid positions are
        those where target_tokens[i] != 0.
        """

        def _pad_int(lst: List[int], length: int) -> List[int]:
            return lst + [0] * (length - len(lst))

        def _pad_float(lst: List[float], length: int) -> List[float]:
            return lst + [0.0] * (length - len(lst))

        target_tokens = torch.tensor(
            [_pad_int(t, max_len) for t in target_tokens_list],
            dtype=torch.long, device=device,
        )  # [B, T]
        old_logp = torch.tensor(
            [_pad_float(lp, max_len) for lp in old_logprobs_list],
            dtype=torch.float32, device=device,
        )  # [B, T]
        advantages = torch.tensor(
            [_pad_float(a, max_len) for a in advantages_list],
            dtype=torch.float32, device=device,
        )  # [B, T]

        # Valid positions: completion tokens have target_tokens != 0.
        # (Prompt positions are padded to 0 by the training loop.)
        valid_mask = target_tokens.ne(0)  # [B, T]

        # Current log probs at each target position.
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        safe_targets = target_tokens.clamp(min=1)   # avoid index 0 in gather; masked out anyway
        current_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # Importance sampling ratio and PPO clip.
        ratio = torch.exp(current_logp - old_logp)                          # [B, T]
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        token_loss = -torch.min(ratio * advantages, clipped * advantages)   # [B, T]

        # Mask out prompt positions and average per sample, then across batch.
        token_loss = token_loss * valid_mask.float()
        valid_counts = valid_mask.float().sum(dim=-1).clamp(min=1)
        per_sample_loss = token_loss.sum(dim=-1) / valid_counts             # [B]
        loss = per_sample_loss.mean()

        return loss, per_sample_loss.detach()

    def _cross_entropy_loss(
        self,
        logits: torch.Tensor,
        batch_tokens: List[List[int]],
        device: torch.device,
    ):
        """Standard next-token cross-entropy loss."""
        batch_size, seq_len, vocab_size = logits.shape

        label_rows: List[List[int]] = []
        for tokens in batch_tokens:
            # Shift: predict token i+1 from position i; mask last position.
            labels = tokens[1:] + [-100]
            pad_len = seq_len - len(labels)
            label_rows.append(labels + [-100] * pad_len)

        labels = torch.tensor(label_rows, dtype=torch.long, device=device)  # [B, T]
        token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(batch_size, seq_len)

        valid_mask = labels.ne(-100).float()
        valid_counts = valid_mask.sum(dim=-1).clamp(min=1)
        per_sample_loss = (token_loss * valid_mask).sum(dim=-1) / valid_counts  # [B]
        loss = per_sample_loss.mean()

        return loss, per_sample_loss.detach()

    # ── Endpoints ────────────────────────────────────────────────────────

    def forward_backward(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
        if not details.data:
            raise ValueError("`data` must contain at least one request.")

        loss_fn = details.loss_fn
        if loss_fn not in ("cross_entropy", "importance_sampling"):
            raise ValueError(
                f"Unsupported loss_fn: {loss_fn!r}. "
                "Supported: 'cross_entropy', 'importance_sampling'."
            )

        with self._lock:
            self._ensure_training_mode("forward_backward")
            t0 = time.perf_counter()
            logger.info(
                "forward_backward: loss_fn=%s, preparing %d sample(s) ...",
                loss_fn, len(details.data),
            )
            self.model.train()

            # Gather per-datum inputs.
            batch_tokens: List[List[int]] = []
            target_tokens_list: List[List[int]] = []
            old_logprobs_list: List[List[float]] = []
            advantages_list: List[List[float]] = []

            for datum in details.data:
                tokens = datum.model_input.to_ints()
                if len(tokens) < 2:
                    raise ValueError("Each datum.model_input must contain at least 2 tokens.")
                batch_tokens.append(tokens)

                if loss_fn == "importance_sampling":
                    tt = datum.loss_fn_inputs.get("target_tokens")
                    lp = datum.loss_fn_inputs.get("logprobs")
                    adv = datum.loss_fn_inputs.get("advantages")
                    if tt is None or lp is None or adv is None:
                        raise ValueError(
                            "importance_sampling requires target_tokens, logprobs, "
                            "and advantages in loss_fn_inputs."
                        )
                    target_tokens_list.append([int(x) for x in tt.data])
                    old_logprobs_list.append([float(x) for x in lp.data])
                    advantages_list.append([float(x) for x in adv.data])

            # Build padded batch tensors.
            pad_token_id = (
                getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", 0)
                or 0
            )
            max_len = max(len(t) for t in batch_tokens)

            input_ids_rows: List[List[int]] = []
            attention_rows: List[List[int]] = []
            for tokens in batch_tokens:
                pad_len = max_len - len(tokens)
                input_ids_rows.append(tokens + [pad_token_id] * pad_len)
                attention_rows.append([1] * len(tokens) + [0] * pad_len)

            device = self._model_device()
            input_ids = torch.tensor(input_ids_rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

            # Forward pass.
            logger.info("forward_backward: running forward pass ...")
            output_tensor = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # Megatron output: [batch_size * seq_len, vocab_size] — reshape to [B, T, V].
            logits = output_tensor.view(batch_size, seq_len, -1)

            # Compute loss.
            if loss_fn == "importance_sampling":
                clip_eps = float((details.loss_fn_config or {}).get("clip_eps", 0.2))
                loss, per_sample_loss = self._importance_sampling_loss(
                    logits, target_tokens_list, old_logprobs_list, advantages_list,
                    max_len=max_len, device=device, clip_eps=clip_eps,
                )
            else:
                loss, per_sample_loss = self._cross_entropy_loss(logits, batch_tokens, device)

            # Backward pass.
            logger.info("forward_backward: running backward pass ...")
            loss.backward()
            self.last_loss = float(loss.detach().cpu())

            loss_fn_outputs = [
                {"loss": TensorData(
                    data=[float(per_sample_loss[i].item())],
                    dtype="float32",
                    shape=[1],
                )}
                for i in range(batch_size)
            ]

            logger.info(
                "forward_backward: done in %.2fs — loss=%.4f, num_samples=%d",
                time.perf_counter() - t0, self.last_loss, len(details.data),
            )
            return ForwardBackwardResult(
                loss_fn_output_type=loss_fn,
                loss_fn_outputs=loss_fn_outputs,
                metrics={"loss": self.last_loss},
            )

    def optim_step(self, details: OptimStepDetails) -> OptimStepResult:
        with self._lock:
            self._ensure_training_mode("optim_step")
            t0 = time.perf_counter()
            logger.info(
                "optim_step: stepping optimizer (current step=%d) ...", self.step
            )

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
                time.perf_counter() - t0, self.step, learning_rate, grad_norm,
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
            # Weight sync to rollout server will be wired here once weight-sync
            # integration is available.
            logger.info("to_inference: weight sync skipped (not yet implemented) ...")
            logger.info(
                "to_inference: done in %.2fs — mode=%s", time.perf_counter() - t0, self.mode
            )
            return ToInferenceResult(mode=self.mode)

    def to_training(self) -> StatusResult:
        with self._lock:
            t0 = time.perf_counter()
            logger.info("to_training: switching from inference to training mode ...")
            self.model.train()
            self.mode = "training"
            logger.info(
                "to_training: done in %.2fs — mode=%s", time.perf_counter() - t0, self.mode
            )
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
                len(prompt_tokens), details.num_samples, effective_max_tokens,
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
                        "logprobs": 1,  # request log prob of the selected token at each position
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                generated_text = choice["text"]
                generated_tokens = self.tokenizer.encode(generated_text, add_special_tokens=False)
                stop_reason = choice.get("finish_reason") or "length"

                # Extract per-token log probs from vLLM response.
                token_logprobs: Optional[List[float]] = None
                logprobs_data = choice.get("logprobs")
                if logprobs_data and "token_logprobs" in logprobs_data:
                    raw = [lp for lp in logprobs_data["token_logprobs"] if lp is not None]
                    if len(raw) == len(generated_tokens):
                        token_logprobs = raw
                    else:
                        logger.warning(
                            "sample: logprob count (%d) != token count (%d); dropping logprobs",
                            len(raw), len(generated_tokens),
                        )

                sequences.append(SampledSequence(
                    tokens=generated_tokens,
                    logprobs=token_logprobs,
                    stop_reason=stop_reason,
                ))

            logger.info(
                "sample: done in %.2fs — %d sequence(s)", time.perf_counter() - t0, len(sequences)
            )
            return SampleResult(sequences=sequences)

    def save_state(self, path: str) -> SaveStateResult:
        with self._lock:
            self._ensure_training_mode("save_state")
            t0 = time.perf_counter()
            logger.info("save_state: saving to %s ...", path)
            ckpt_dir = Path(path)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            logger.info("save_state: exporting Megatron weights to HF format ...")
            self.bridge.save_weights([self.model], str(ckpt_dir))
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
