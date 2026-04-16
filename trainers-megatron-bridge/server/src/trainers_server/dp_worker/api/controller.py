"""Megatron-bridge-based RL training controller.

All collective methods (forward_backward, optim_step, to_inference, save_state)
must be called on ALL distributed ranks simultaneously. Rank 0 drives requests
from the HTTP layer; non-zero ranks execute the corresponding _execute_* method
via the worker loop in main.py.
"""

import logging
import os
import pickle
import socket
import subprocess
import time
import warnings
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F

# megatron imports are deferred to RLController.__init__ so this module can be
# imported without the full megatron/transformer-engine stack (e.g. in tests
# that mock the controller).

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

from .models import RLControllerConfig, StatusResult

logger = logging.getLogger(__name__)

# Op codes broadcast from rank 0 to worker ranks.
OP_FORWARD_BACKWARD = 1
OP_OPTIM_STEP = 2
OP_TO_INFERENCE = 3
OP_TO_TRAINING = 4
OP_SAVE_STATE = 5
OP_EXIT = 255

_ROLLOUT_STARTUP_TIMEOUT = 300
_ROLLOUT_RETRY_INTERVAL = 5
# Path used for HF-format weight export during to_inference().
_WEIGHT_SYNC_PATH = "/tmp/megatron_weight_sync"


# ── helpers ───────────────────────────────────────────────────────────


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _comm_device() -> Optional[torch.device]:
    """Return the device tensors must live on for the current process group.

    NCCL (GPU-to-GPU) requires CUDA tensors; gloo (CPU) accepts CPU tensors.
    Returns None when dist is not initialised (single-process path).
    """
    if not _is_distributed():
        return None
    backend = dist.get_backend()
    if backend == "nccl":
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _broadcast_tensor(t: torch.Tensor) -> None:
    """Broadcast *t* in-place from rank 0 to all ranks.

    Handles NCCL (requires CUDA tensors) transparently by moving to CUDA,
    broadcasting, and copying the result back into the original CPU tensor.
    """
    if not _is_distributed():
        return
    dev = _comm_device()
    if dev is not None and t.device != dev:
        t_dev = t.to(dev)
        dist.broadcast(t_dev, src=0)
        t.copy_(t_dev)  # write result back into the caller's tensor
    else:
        dist.broadcast(t, src=0)


def _broadcast_bytes(data: Optional[bytes]) -> bytes:
    """Broadcast a byte buffer from rank 0 to all ranks.

    On rank 0, *data* must be the bytes to send.
    On other ranks, *data* is ignored; the received bytes are returned.
    All ranks must call this at the same point in execution.
    """
    if not _is_distributed():
        return data  # type: ignore[return-value]

    dev = _comm_device() or torch.device("cpu")
    rank = _rank()
    size_t = torch.tensor([len(data) if rank == 0 else 0], dtype=torch.int64, device=dev)
    dist.broadcast(size_t, src=0)
    n = int(size_t.item())

    if rank == 0:
        buf = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(dev)  # type: ignore[arg-type]
    else:
        buf = torch.zeros(n, dtype=torch.uint8, device=dev)
    dist.broadcast(buf, src=0)

    if rank != 0:
        data = bytes(buf.cpu().numpy())
    return data  # type: ignore[return-value]


def _barrier() -> None:
    if _is_distributed():
        dist.barrier()


# ── controller ────────────────────────────────────────────────────────


class RLController:
    """Megatron-based RL controller supporting multi-GPU tensor/pipeline parallelism.

    Lifecycle
    ---------
    1. Instantiate on *every* rank inside `mp.spawn` after torch.distributed is
       initialized.
    2. Rank 0 runs the FastAPI HTTP server; other ranks run `worker_loop(controller)`.
    3. Each public method on rank 0 broadcasts an op-code + payload to workers,
       which execute the matching _execute_* method collectively.
    4. Call `close()` (or let `__del__` handle it) when done.
    """

    def __init__(self, config: RLControllerConfig) -> None:
        self.config = config
        self._lock = RLock()
        self.mode: str = "training"
        self.step: int = 0
        self.last_loss: Optional[float] = None
        self._closed: bool = False
        self._rollout_process: Optional[subprocess.Popen] = None
        self._tokenizer = None
        self._pad_token_id: Optional[int] = None

        # Find a rollout port on rank 0 and broadcast to all ranks.
        # _broadcast_tensor handles NCCL's requirement for CUDA tensors.
        port_t = torch.tensor([self._find_free_port() if _rank() == 0 else 0], dtype=torch.int64)
        _broadcast_tensor(port_t)
        self._rollout_port: int = int(port_t.item())

        # ── Load Megatron model via megatron-bridge ──────────────────
        from megatron.bridge import AutoBridge  # noqa: PLC0415
        from megatron.core.distributed import DistributedDataParallelConfig  # noqa: PLC0415
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer  # noqa: PLC0415

        logger.info("Loading model %s via megatron-bridge ...", config.model_id)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bridge = AutoBridge.from_hf_pretrained(config.model_id, torch_dtype=torch.bfloat16)
            provider = bridge.to_megatron_provider()
            provider.tensor_model_parallel_size = config.training.tensor_parallel_size
            provider.pipeline_model_parallel_size = config.training.pipeline_parallel_size
            provider.params_dtype = torch.bfloat16
            provider.pipeline_dtype = torch.bfloat16
            # gradient_accumulation_fusion requires the fused_weight_gradient_mlp_cuda
            # APEX extension which may not be installed; disable to avoid RuntimeError.
            provider.gradient_accumulation_fusion = False
            provider.finalize()
            # DDP config required by get_megatron_optimizer (even with world_size=1).
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                use_distributed_optimizer=True,
            )
            # provide_distributed_model is collective — all ranks participate.
            self.model: list = provider.provide_distributed_model(
                wrap_with_ddp=True,
                ddp_config=ddp_config,
            )
        self.bridge = bridge
        logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

        # ── Megatron optimizer ────────────────────────────────────────
        opt_config = OptimizerConfig(
            optimizer="adam",
            lr=1e-4,          # overridden per optim_step
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=0.0,
            use_distributed_optimizer=True,
            bf16=True,
        )
        self.optimizer = get_megatron_optimizer(opt_config, self.model)
        self._opt_config = opt_config

        # ── vLLM rollout server (rank 0 only) ─────────────────────────
        if _rank() == 0:
            logger.info("Launching vLLM rollout server on GPUs=%s ...", config.inference.gpus)
            self._launch_rollout(config.model_id)
            self._wait_for_rollout()

        _barrier()
        logger.info("RLController ready — rank=%d, mode=%s", _rank(), self.mode)

    # ── Port / subprocess helpers ──────────────────────────────────────

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(s.getsockname()[1])

    def _launch_rollout(self, model_path: str) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.config.inference.gpus)
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", "rollout",
            "--tensor-parallel-size", str(self.config.inference.tensor_parallel_size),
            "--host", "127.0.0.1",
            "--port", str(self._rollout_port),
            "--gpu-memory-utilization", str(self.config.inference.gpu_memory_utilization),
            "--dtype", "bfloat16",
            "--max-model-len", str(max(self.config.training.max_length, 4096)),
            "--enforce-eager",
            "--disable-log-requests",
        ]
        logger.info("vLLM cmd: %s", " ".join(cmd))
        self._rollout_process = subprocess.Popen(cmd, env=env)

    def _wait_for_rollout(self) -> None:
        deadline = time.time() + _ROLLOUT_STARTUP_TIMEOUT
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            if self._rollout_process is not None and self._rollout_process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup (code={self._rollout_process.returncode})"
                )
            try:
                r = requests.get(
                    f"http://127.0.0.1:{self._rollout_port}/health",
                    timeout=float(_ROLLOUT_RETRY_INTERVAL),
                )
                if r.status_code == 200:
                    logger.info("vLLM ready after %d attempts", attempt)
                    return
            except Exception as exc:
                logger.debug("vLLM not ready (attempt %d): %s", attempt, exc)
            time.sleep(_ROLLOUT_RETRY_INTERVAL)
        raise RuntimeError(f"vLLM did not become ready within {_ROLLOUT_STARTUP_TIMEOUT}s")

    def _kill_rollout(self) -> None:
        proc = self._rollout_process
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self._rollout_process = None

    # ── Tokenizer (lazy-loaded, rank 0 only for sampling) ─────────────

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._pad_token_id = (
                self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id
                or 0
            )

    # ── Batch construction ─────────────────────────────────────────────

    def _build_batch(self, data: List[Datum]) -> Dict[str, torch.Tensor]:
        """Build padded tensors from a list of Datum objects.

        Returns a dict with keys:
          input_ids   (B, S)     — input token IDs (padded)
          labels      (B, S)     — next-token targets; 0 for padding positions
          loss_mask   (B, S)     — 1.0 for valid next-token positions, 0.0 elsewhere
          position_ids(B, S)     — position indices (0 for padding)
          rewards     (B,)       — per-sample scalar rewards
        """
        self._ensure_tokenizer()

        batch_tokens: List[List[int]] = []
        rewards: List[float] = []
        for datum in data:
            tokens = datum.model_input.to_ints()
            if len(tokens) < 2:
                raise ValueError("Each datum.model_input must contain at least 2 tokens.")
            batch_tokens.append(tokens)

            # Extract scalar reward from loss_fn_inputs["reward"].
            reward_td = datum.loss_fn_inputs.get("reward")
            if reward_td is None:
                rewards.append(1.0)
            else:
                val = reward_td.data
                while isinstance(val, list):
                    val = val[0] if val else 0
                rewards.append(float(val))

        B = len(batch_tokens)
        S = max(len(t) for t in batch_tokens)
        pad = self._pad_token_id or 0

        input_ids = torch.full((B, S), pad, dtype=torch.long)
        # labels: next-token targets — pad with 0 (masked by loss_mask)
        labels = torch.zeros(B, S, dtype=torch.long)
        loss_mask = torch.zeros(B, S, dtype=torch.float)
        position_ids = torch.zeros(B, S, dtype=torch.long)

        for i, tokens in enumerate(batch_tokens):
            L = len(tokens)
            t = torch.tensor(tokens, dtype=torch.long)
            input_ids[i, :L] = t
            # labels[i, s] = tokens[s+1] for s in 0..L-2; 0 for s >= L-1
            if L >= 2:
                labels[i, : L - 1] = t[1:]
                loss_mask[i, : L - 1] = 1.0
            position_ids[i, :L] = torch.arange(L, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "rewards": torch.tensor(rewards, dtype=torch.float),
        }

    # ── Public API (rank 0 entry points) ──────────────────────────────

    def forward_backward(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
        with self._lock:
            if details.loss_fn != "cross_entropy":
                raise ValueError(
                    f"Unsupported loss_fn={details.loss_fn!r}. Only 'cross_entropy' is supported."
                )
            self._ensure_training_mode("forward_backward")

            # Signal workers.
            op_t = torch.tensor([OP_FORWARD_BACKWARD], dtype=torch.int32)
            _broadcast_tensor(op_t)

            # Broadcast serialized details to all ranks.
            payload = pickle.dumps(details.model_dump())
            _broadcast_bytes(payload)

            result = self._execute_forward_backward(details)
            assert result is not None
            return result

    def optim_step(self, details: OptimStepDetails) -> OptimStepResult:
        with self._lock:
            self._ensure_training_mode("optim_step")

            op_t = torch.tensor([OP_OPTIM_STEP], dtype=torch.int32)
            _broadcast_tensor(op_t)

            payload = pickle.dumps(details.model_dump())
            _broadcast_bytes(payload)

            result = self._execute_optim_step(details)
            assert result is not None
            return result

    def to_inference(self) -> ToInferenceResult:
        with self._lock:
            t0 = time.perf_counter()
            op_t = torch.tensor([OP_TO_INFERENCE], dtype=torch.int32)
            _broadcast_tensor(op_t)

            self._execute_to_inference()

            self.mode = "inference"
            logger.info("to_inference: complete in %.1fs", time.perf_counter() - t0)
            return ToInferenceResult(mode=self.mode)

    def to_training(self) -> StatusResult:
        with self._lock:
            op_t = torch.tensor([OP_TO_TRAINING], dtype=torch.int32)
            _broadcast_tensor(op_t)

            self.mode = "training"
            return self.get_status()

    def sample(self, details: SampleDetails) -> SampleResult:
        """Generate samples from the vLLM rollout server.  Rank-0 only."""
        with self._lock:
            if self.mode != "inference":
                raise RuntimeError("sample() is only valid in inference mode.")
            if _rank() != 0:
                return SampleResult(sequences=[])

            t0 = time.perf_counter()
            self._ensure_tokenizer()

            prompt_tokens = details.prompt.to_ints()
            params = details.sampling_params
            max_tokens = params.max_tokens or 128

            payload: dict = {
                "model": "rollout",
                "prompt": prompt_tokens,  # vLLM accepts a list of token IDs
                "max_tokens": max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "n": details.num_samples,
                "stream": False,
            }
            if params.stop:
                payload["stop"] = (
                    params.stop if isinstance(params.stop, list) else [params.stop]
                )

            resp = requests.post(
                f"http://127.0.0.1:{self._rollout_port}/v1/completions",
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()

            sequences: List[SampledSequence] = []
            for choice in data.get("choices", []):
                text = choice.get("text", "")
                tokens = self._tokenizer.encode(text, add_special_tokens=False)
                sequences.append(
                    SampledSequence(
                        tokens=tokens,
                        stop_reason=choice.get("finish_reason", "length"),
                    )
                )

            logger.info(
                "sample: %.2fs — %d sequences", time.perf_counter() - t0, len(sequences)
            )
            return SampleResult(sequences=sequences)

    def save_state(self, path: str) -> SaveStateResult:
        with self._lock:
            op_t = torch.tensor([OP_SAVE_STATE], dtype=torch.int32)
            _broadcast_tensor(op_t)

            payload = pickle.dumps(path)
            _broadcast_bytes(payload)

            self._execute_save_state(path)
            return SaveStateResult(mode=self.mode)

    def get_status(self) -> StatusResult:
        with self._lock:
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_mem: Dict[str, int] = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem[f"cuda:{i}"] = int(torch.cuda.memory_allocated(i))
            return StatusResult(
                mode=self.mode,
                step=self.step,
                model_id=self.config.model_id,
                device=device,
                last_loss=self.last_loss,
                grad_norm=None,
                gpu_memory=gpu_mem,
            )

    # ── Collective _execute_* methods (called on all ranks) ───────────

    def _execute_forward_backward(
        self, details: ForwardBackwardDetails
    ) -> Optional[ForwardBackwardResult]:
        """Forward + backward on all TP/PP ranks.  Returns result only on rank 0."""
        t0 = time.perf_counter()
        batch = self._build_batch(details.data)

        device = torch.cuda.current_device()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        rewards = batch["rewards"].to(device)
        position_ids = batch["position_ids"].to(device)

        B, S = input_ids.shape

        # ── Forward pass ─────────────────────────────────────────────
        # For PP=1 the model list has one element.
        # When labels=(B,S) are provided, compute_language_model_loss is called
        # inside GPTModel, which runs vocab_parallel_cross_entropy (handles TP
        # vocab split) and returns per-token losses with shape (B, S).
        gpt_model = self.model[0]
        gpt_model.train()

        output_tensor = gpt_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,   # causal mask generated internally
            labels=labels,
        )

        # output_tensor: (B, S) per-token cross-entropy losses on the last PP
        # stage (post_process=True). Intermediate PP stages return hidden
        # activations passed via pipeline communicators — we skip manual loss
        # computation there. For PP=1, output is always (B, S).
        if output_tensor is not None and output_tensor.shape == (B, S):
            per_token = output_tensor.float() * loss_mask   # zero out padding
            valid_counts = loss_mask.sum(dim=-1).clamp(min=1)   # (B,)
            per_sample = per_token.sum(dim=-1) / valid_counts    # (B,)
            loss = (per_sample * rewards).mean()
        elif output_tensor is not None:
            # Unexpected shape (e.g., intermediate PP activation) — fall back
            # to a zero loss so backward is a no-op on this rank.
            loss = torch.zeros(1, device=device, requires_grad=True)
            loss = loss * output_tensor.sum() * 0  # keep graph connected
        else:
            loss = torch.zeros(1, device=device, requires_grad=True)

        # ── Backward pass ─────────────────────────────────────────────
        loss.backward()
        loss_scalar = float(loss.detach().cpu())

        if _rank() == 0:
            self.last_loss = loss_scalar
            logger.info(
                "forward_backward: %.2fs — loss=%.4f, n_samples=%d",
                time.perf_counter() - t0,
                loss_scalar,
                len(details.data),
            )
            return ForwardBackwardResult(
                loss_fn_output_type="per_token_logprobs",
                loss_fn_outputs=[],
                metrics={"loss": loss_scalar},
            )
        return None

    def _execute_optim_step(
        self, details: OptimStepDetails
    ) -> Optional[OptimStepResult]:
        """Optimizer step on all ranks. Returns result only on rank 0."""
        t0 = time.perf_counter()
        params = details.adam_params

        # Update learning rate and other hyperparameters.
        for pg in self.optimizer.param_groups:
            pg["lr"] = params.learning_rate
            pg["betas"] = (params.beta1, params.beta2)
            pg["eps"] = params.eps
            pg["weight_decay"] = params.weight_decay

        self._opt_config.clip_grad = params.grad_clip_norm if params.grad_clip_norm > 0 else 0.0

        update_successful, grad_norm, num_zeros = self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1

        if _rank() == 0:
            lr = float(params.learning_rate)
            gn = float(grad_norm) if grad_norm is not None else 0.0
            logger.info(
                "optim_step: %.2fs — step=%d, lr=%.2e, grad_norm=%.4f, ok=%s",
                time.perf_counter() - t0,
                self.step,
                lr,
                gn,
                update_successful,
            )
            return OptimStepResult(
                metrics={
                    "step": float(self.step),
                    "learning_rate": lr,
                    "lr": lr,
                    "grad_norm": gn,
                }
            )
        return None

    def _execute_to_inference(self) -> None:
        """Export weights to HF format and reload vLLM.  All ranks participate."""
        logger.info("rank %d: exporting Megatron weights to HF format ...", _rank())

        # save_hf_pretrained is collective — all ranks must call it.
        self.bridge.save_hf_pretrained(
            self.model,
            _WEIGHT_SYNC_PATH,
            show_progress=(_rank() == 0),
        )

        if _rank() == 0:
            logger.info("Reloading vLLM from %s ...", _WEIGHT_SYNC_PATH)
            self._kill_rollout()
            self._launch_rollout(_WEIGHT_SYNC_PATH)
            self._wait_for_rollout()

        _barrier()

    def _execute_save_state(self, path: str) -> None:
        """Save HF-format checkpoint + trainer state.  All ranks participate."""
        logger.info("rank %d: saving checkpoint to %s ...", _rank(), path)
        self.bridge.save_hf_pretrained(
            self.model,
            path,
            show_progress=(_rank() == 0),
        )
        if _rank() == 0:
            trainer_state = {
                "step": self.step,
                "mode": self.mode,
                "last_loss": self.last_loss,
                "config": self.config.model_dump(),
            }
            torch.save(trainer_state, Path(path) / "trainer_state.pt")
        _barrier()
        logger.info("rank %d: checkpoint saved", _rank())

    # ── Utility ───────────────────────────────────────────────────────

    def _ensure_training_mode(self, caller: str) -> None:
        if self.mode != "training":
            logger.info("%s: switching to training mode", caller)
            self.mode = "training"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if _rank() == 0:
                self._kill_rollout()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── Worker loop (run on ranks 1 .. world_size-1) ──────────────────────


def worker_loop(controller: RLController) -> None:
    """Block and respond to collective ops broadcast by rank 0.

    Non-rank-0 processes call this after `RLController.__init__` returns.
    The loop exits when it receives OP_EXIT.
    """
    rank = _rank()
    logger.info("Worker rank %d entering dispatch loop", rank)

    while True:
        op_t = torch.zeros(1, dtype=torch.int32)
        _broadcast_tensor(op_t)
        op = int(op_t.item())

        if op == OP_EXIT:
            logger.info("Worker rank %d: EXIT received, stopping", rank)
            break

        elif op == OP_FORWARD_BACKWARD:
            raw = _broadcast_bytes(None)
            details = ForwardBackwardDetails.model_validate(pickle.loads(raw))
            controller._execute_forward_backward(details)

        elif op == OP_OPTIM_STEP:
            raw = _broadcast_bytes(None)
            details = OptimStepDetails.model_validate(pickle.loads(raw))
            controller._execute_optim_step(details)

        elif op == OP_TO_INFERENCE:
            controller._execute_to_inference()

        elif op == OP_TO_TRAINING:
            controller.mode = "training"

        elif op == OP_SAVE_STATE:
            raw = _broadcast_bytes(None)
            path: str = pickle.loads(raw)
            controller._execute_save_state(path)

        else:
            logger.warning("Worker rank %d: unknown op code %d — ignored", rank, op)
