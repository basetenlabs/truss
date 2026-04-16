"""
Megatron-backed RL controller.

Why Megatron (not plain HF)?
-----------------------------
Weight sync to vLLM requires ``GPTBridge.export_weights(mg_models)``, which:

  * Expects ``trainer.unwrapped_models`` to be Megatron ``GPTModel`` instances
    (parameters are TP-sharded across the tensor-parallel group).
  * Uses ``mpu.get_tensor_model_parallel_group()`` to gather TP shards before
    converting to HF parameter names/shapes for vLLM.

A plain HF model cannot satisfy either requirement.  This controller uses
``MegatronSft.prepare_trainer()`` to initialise Megatron (TP/PP process groups,
model, optimizer) and then drives training **step-by-step** via HTTP — the same
external interface as the HF-backed ``RLController``.

Key references after ``__init__``:
  - ``self.unwrapped_models``  — Megatron GPTModel list (pass to
                                  ``MegatronWeightWriter`` or call
                                  ``trainer.bridge.export_weights`` directly)
  - ``self.trainer``           — full ``MegatronTrainer`` for advanced use
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import socket
import time
import atexit
from pathlib import Path
from threading import RLock
from typing import List, Optional, Tuple

import requests
import torch

# Heavy swift.megatron imports are deferred to __init__ / method bodies so
# that the module can be imported in test environments where ms-swift is
# replaced by lightweight stubs (conftest.py).  The top-level swift.* stubs
# only cover the HF path; they do not provide swift.megatron.*.
from swift.arguments import RolloutArguments
from swift.rlhf_trainers.utils import FlattenedTensorBucket
from swift.rlhf_trainers.vllm_client import VLLMClient
from swift.pipelines.infer.rollout import rollout_main

from trainers_server.dp_worker.distributed import (
    is_rank_zero,
    is_distributed,
    get_rank,
)
from trainers_server.shared.models import (
    Datum,
    ForwardBackwardDetails,
    ForwardBackwardResult,
    OptimStepDetails,
    OptimStepResult,
    SampleDetails,
    SampleResult,
    SampledSequence,
    SaveStateResult,
    ToInferenceResult,
)
from .models import RLControllerConfig, StatusResult

logger = logging.getLogger(__name__)

_ROLLOUT_STARTUP_TIMEOUT_SECONDS = 300
_ROLLOUT_RETRY_INTERVAL_SECONDS = 5


def _rollout_server_entry(config_data: dict, rollout_port: int) -> None:
    os.setsid()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGTERM)
    except Exception:
        pass

    config = RLControllerConfig(**config_data)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.inference.gpus)
    rollout_max_model_len = max(config.training.max_length, 4096)

    args = RolloutArguments(
        model=config.model_id,
        host="127.0.0.1",
        port=rollout_port,
        use_hf=True,
        torch_dtype="bfloat16",
        max_length=rollout_max_model_len,
        infer_backend="vllm",
        vllm_tensor_parallel_size=config.inference.tensor_parallel_size,
        vllm_data_parallel_size=1,
        vllm_gpu_memory_utilization=config.inference.gpu_memory_utilization,
        vllm_max_model_len=rollout_max_model_len,
        vllm_enforce_eager=True,
        vllm_use_async_engine=False,
        vllm_engine_kwargs={"load_format": "auto"},
        log_level="info",
    )
    rollout_main(args)


class MegatronRLController:
    """RL controller backed by Megatron-Core (TP/PP aware).

    Parameters
    ----------
    config:
        Standard ``RLControllerConfig``.  ``training.tensor_parallel_size`` and
        ``training.pipeline_parallel_size`` are forwarded to Megatron.

    Usage
    -----
    After construction the controller is ready for the standard HTTP API
    (``forward_backward``, ``optim_step``, ``to_inference``, …).

    For Will's ``MegatronWeightWriter``::

        writer = MegatronWeightWriter(
            model=controller.unwrapped_models,
            vllm_urls=[vllm_url],
            model_path=config.model_id,
        )
        writer.sync(version=step)
    """

    def __init__(self, config: RLControllerConfig) -> None:
        # Lazy imports: defer swift.megatron.* until __init__ runs so the module
        # can be imported in test environments with lightweight stubs.
        from swift.megatron.arguments import MegatronSftArguments  # noqa: F401 (used below)
        from swift.megatron.pipelines.train.sft import MegatronSft  # noqa: F401

        init_t0 = time.perf_counter()
        logger.info("MegatronRLController.__init__ starting for model=%s", config.model_id)
        self.config = config
        self._lock = RLock()
        self.mode = "training"
        self.step = 0
        self.last_loss: Optional[float] = None
        self._closed = False
        self._rollout_process: Optional[multiprocessing.Process] = None
        self.vllm_client: Optional[VLLMClient] = None
        self._communicator_ready = False
        self._rollout_port: int = 0
        self._rollout_group_port: int = 0

        # ── Megatron initialisation ─────────────────────────────────────────
        # MegatronSft creates the trainer (which patches Megatron's
        # setup_model_and_optimizer to capture unwrapped_models), but it does
        # NOT load the model yet — loading happens inside pretrain() when
        # setup_model_and_optimizer is actually called.
        #
        # We drive that setup by calling trainer.train() with the Megatron
        # training loop replaced by a no-op, so only the model/optimizer
        # setup runs (steps 1-2 of pretrain) and we exit before any real
        # training iterations.
        logger.info("Initialising MegatronSft (TP=%d, PP=%d) …",
                    config.training.tensor_parallel_size,
                    config.training.pipeline_parallel_size)
        sft_args = self._build_megatron_sft_args(config)
        self._sft = MegatronSft(sft_args)
        self.trainer = self._sft.trainer  # BaseMegatronTrainer / MegatronTrainer
        self.processor = self._sft.processor

        logger.info("Triggering Megatron model/optimizer setup …")
        self._setup_megatron_model()

        # ── Key references ──────────────────────────────────────────────────
        # unwrapped_models: list of Megatron GPTModel (one per PP stage).
        # Pass these to MegatronWeightWriter or call trainer.bridge.export_weights.
        self.unwrapped_models = self.trainer.unwrapped_models

        logger.info("Megatron model loaded; unwrapped_models=%d PP stage(s), TP=%d",
                    len(self.unwrapped_models),
                    config.training.tensor_parallel_size)

        # ── Rollout server (rank 0 only) ────────────────────────────────────
        if not is_distributed() or is_rank_zero():
            self._rollout_port = self._find_free_port()
            self._rollout_group_port = self._find_free_port()
            self._rollout_max_model_len = max(config.training.max_length, 4096)
            logger.info("Launching rollout server on GPUs=%s …", config.inference.gpus)
            self._launch_rollout_server()
            logger.info("Connecting VLLMClient to rollout server …")
            self._init_vllm_client()

        logger.info("MegatronRLController.__init__ complete in %.1fs", time.perf_counter() - init_t0)

    def _setup_megatron_model(self) -> None:
        """Trigger Megatron's model/optimizer setup without running training.

        MegatronTrainer.train(dataset, val_dataset, collator) calls
        megatron.pretrain(), which:
          1. initialize_megatron() — TP/PP process groups, global args
          2. setup_model_and_optimizer() — loads model+optimizer, patched by
             swift to populate trainer.unwrapped_models / wrapped_models
          3. data iterators — we short-circuit with dummy iterators
          4. training loop — we replace with a no-op

        After step 2, self.trainer.unwrapped_models is populated and the
        controller is ready for step-by-step forward/backward via HTTP.
        """
        import megatron.training.training as mg_training

        class _SetupDone(Exception):
            pass

        _orig_train = mg_training.train
        _orig_data = mg_training.build_train_valid_test_data_iterators

        _trainer = self.trainer

        def _noop_train(forward_step_func, model, optimizer, opt_param_scheduler, *_a, **_kw):
            # Capture the optimizer before short-circuiting the training loop.
            _trainer.optimizer = optimizer
            _trainer.opt_param_scheduler = opt_param_scheduler
            raise _SetupDone()

        def _dummy_data(*_a, **_kw):
            # Set flags required by pretrain() after dataloader setup.
            # Keep do_train=True so pretrain() calls train(), where _noop_train
            # captures the optimizer before short-circuiting the loop.
            from megatron.training import get_args
            try:
                _args = get_args()
                _args.do_train = True
                _args.do_valid = False
                _args.do_test = False
            except Exception:
                pass
            return None, None, None

        mg_training.train = _noop_train
        mg_training.build_train_valid_test_data_iterators = _dummy_data
        try:
            # Pass dummy dataset and collator — they're only needed for the
            # training loop which we intercept before it starts.
            data_collator = self._sft._get_data_collator()
            self.trainer.train(
                train_dataset=None,
                val_dataset=None,
                data_collator=data_collator,
            )
        except _SetupDone:
            pass  # model+optimizer set up; training loop was skipped
        finally:
            mg_training.train = _orig_train
            mg_training.build_train_valid_test_data_iterators = _orig_data

    # ── Megatron argument builder ────────────────────────────────────────────

    @staticmethod
    def _build_megatron_sft_args(config: RLControllerConfig):
        """Map ``RLControllerConfig`` fields to ``MegatronSftArguments``.

        ``train_iters`` is set to a very large number because training is driven
        step-by-step via HTTP — the Megatron training loop never actually runs.

        A tiny placeholder ``dataset`` entry is required to pass argument
        validation; it is never consumed because we bypass the Megatron
        training loop and drive forward/backward passes directly via HTTP.
        """
        from swift.megatron.arguments import MegatronSftArguments
        return MegatronSftArguments(
            model=config.model_id,
            tuner_type="full",
            tensor_model_parallel_size=config.training.tensor_parallel_size,
            pipeline_model_parallel_size=config.training.pipeline_parallel_size,
            max_length=config.training.max_length,
            # micro_batch_size is used for Megatron's internal buffer sizing.
            # We override it per-call in forward_backward_func.
            micro_batch_size=1,
            # Effectively unlimited — we drive steps via HTTP.
            train_iters=int(1e12),
            save_strategy="no",
            task_type="causal_lm",
            # lr=0: we set the real lr per optim_step via adam_params.
            lr=0.0,
            # Do not start vLLM inside the trainer — we manage it separately.
            use_vllm=False,
            # Required to pass argument validation; never actually consumed
            # because we bypass MegatronSft's training loop.
            dataset=["alpaca#1"],
        )

    # ── Rollout server lifecycle ────────────────────────────────────────────

    def _launch_rollout_server(self) -> None:
        mp_ctx = multiprocessing.get_context("spawn")
        process = mp_ctx.Process(
            target=_rollout_server_entry,
            args=(self.config.model_dump(), self._rollout_port),
        )
        process.start()
        self._rollout_process = process

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(s.getsockname()[1])

    def _init_vllm_client(self) -> None:
        deadline = time.time() + _ROLLOUT_STARTUP_TIMEOUT_SECONDS
        attempt = 0
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            attempt += 1
            if self._rollout_process is not None and not self._rollout_process.is_alive():
                raise RuntimeError(
                    f"Rollout process exited during startup (exit_code={self._rollout_process.exitcode})."
                )
            logger.info("Waiting for rollout server (attempt %d) …", attempt)
            try:
                r = requests.get(f"http://127.0.0.1:{self._rollout_port}/health/",
                                 timeout=_ROLLOUT_RETRY_INTERVAL_SECONDS)
                if r.status_code != 200:
                    raise RuntimeError(f"Health returned {r.status_code}")
                self.vllm_client = VLLMClient(
                    hosts=["127.0.0.1"],
                    server_ports=[self._rollout_port],
                    group_ports=[self._rollout_group_port],
                    connection_timeout=float(_ROLLOUT_RETRY_INTERVAL_SECONDS),
                )
                logger.info("VLLMClient connected.")
                return
            except Exception as e:
                last_error = e
                logger.info("Rollout not ready: %s", e)
                time.sleep(_ROLLOUT_RETRY_INTERVAL_SECONDS)
        raise RuntimeError(f"Timed out waiting for rollout server. Last error: {last_error}")

    def _ensure_communicator_ready(self) -> None:
        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")
        if self._communicator_ready:
            return
        # Use the first unwrapped model's device as the training device.
        device = next(self.unwrapped_models[0].parameters()).device
        logger.info("Initialising rollout communicator on device %s …", device)
        self.vllm_client.init_communicator(device=str(device))
        self._communicator_ready = True

    # ── Batch conversion ─────────────────────────────────────────────────────

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

    def _to_megatron_batch(self, details: ForwardBackwardDetails) -> dict:
        """Convert ``ForwardBackwardDetails`` to Megatron's expected batch format.

        Megatron's ``task_type='causal_lm'`` path in ``get_batch_on_this_tp_rank``
        does ``labels = torch.roll(labels, -1)`` so that ``labels[t] = tokens[t+1]``
        (next-token prediction).  We therefore pass ``labels = input_ids`` (same
        as inputs) and set padding positions to -100 so they are masked out by
        the loss function.

        ``loss_scale`` receives the per-sample RL reward, broadcast across all
        valid token positions.  It is also rolled by -1 in Megatron, keeping it
        aligned with ``labels``.
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_id: int = getattr(tokenizer, "pad_token_id", None) or 0

        batch_tokens: List[List[int]] = []
        rewards: List[float] = []
        for datum in details.data:
            tokens = datum.model_input.to_ints()
            if len(tokens) < 2:
                raise ValueError("Each datum.model_input must contain at least 2 tokens.")
            batch_tokens.append(tokens)
            rewards.append(self._extract_reward(datum))

        B = len(batch_tokens)

        # With padding_free=True, swift expects packed sequences: all tokens
        # concatenated into a single [1, total_tokens] tensor.  Position IDs
        # restart from 0 for each sequence so that get_packed_seq_params() can
        # derive cu_seqlens by finding positions where pos_id == 0.
        all_input: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_pos: List[torch.Tensor] = []
        all_scale: List[torch.Tensor] = []

        for tokens, reward in zip(batch_tokens, rewards):
            n = len(tokens)
            t = torch.tensor(tokens, dtype=torch.long)
            all_input.append(t)
            all_labels.append(t)                   # rolled left by Megatron → next-token targets
            all_pos.append(torch.arange(n, dtype=torch.long))
            all_scale.append(torch.full((n,), reward, dtype=torch.float32))

        # Shape: [1, total_tokens]
        input_ids  = torch.cat(all_input).unsqueeze(0)
        labels     = torch.cat(all_labels).unsqueeze(0)
        position_ids = torch.cat(all_pos).unsqueeze(0)
        loss_scale = torch.cat(all_scale).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "loss_scale": loss_scale,
            "num_samples": B,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def forward_backward(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
        if not details.data:
            raise ValueError("`data` must contain at least one request.")
        with self._lock:
            self._ensure_training_mode("forward_backward")
            if details.loss_fn != "cross_entropy":
                raise ValueError(
                    f"Unsupported loss_fn: {details.loss_fn!r}. Only 'cross_entropy' is supported."
                )
            return self._forward_backward_impl(details)

    def optim_step(self, details: OptimStepDetails) -> OptimStepResult:
        with self._lock:
            self._ensure_training_mode("optim_step")
            return self._optim_step_impl(details)

    def to_inference(self) -> ToInferenceResult:
        with self._lock:
            return self._to_inference_impl()

    def to_training(self) -> StatusResult:
        with self._lock:
            self._to_training_impl()
            return self.get_status()

    def sample(self, details: SampleDetails) -> SampleResult:
        with self._lock:
            if self.mode != "inference":
                raise RuntimeError("sample is only allowed in inference mode.")
            return self._sample_impl(details)

    def save_state(self, path: str) -> SaveStateResult:
        with self._lock:
            self._ensure_training_mode("save_state")
            return self._save_state_impl(path)

    # ── Implementation methods ───────────────────────────────────────────────

    def _forward_backward_impl(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
        from megatron.core.pipeline_parallel import get_forward_backward_func

        t0 = time.perf_counter()
        logger.info("forward_backward: %d sample(s) …", len(details.data))

        batch = self._to_megatron_batch(details)
        B, T = batch["input_ids"].shape

        forward_backward_func = get_forward_backward_func()

        # The RerunStateMachine expects to be in RUNNING state (set by
        # megatron's train_step wrapper) when validate_result() is called from
        # inside the loss function.  Since we call forward_backward_func
        # directly we must disable it for the duration of the call.
        try:
            from megatron.core.rerun_state_machine import (
                get_rerun_state_machine,
                RerunMode,
            )
            _rsm = get_rerun_state_machine()
            _prev_mode = _rsm.get_mode()
            _rsm.set_mode(RerunMode.DISABLED)
        except Exception:
            _rsm = None
            _prev_mode = None

        try:
            loss_dicts = forward_backward_func(
                forward_step_func=self.trainer.forward_step,
                data_iterator=iter([batch]),
                model=self.trainer.wrapped_models,
                num_microbatches=1,
                seq_length=T,
                micro_batch_size=B,
                forward_only=False,
            )
        finally:
            if _rsm is not None and _prev_mode is not None:
                _rsm.set_mode(_prev_mode)

        # Megatron's DDP accumulates gradients in param.main_grad rather than
        # param.grad (which it sets to None via backward hooks).  Expose a grad
        # tensor on each parameter so callers that inspect param.grad work.
        # main_grad is fp32 (accumulate_allreduce_grads_in_fp32=True) so we
        # cast to the parameter's dtype to satisfy PyTorch's dtype check.
        for m in self.unwrapped_models:
            for p in m.parameters():
                if p.grad is None and hasattr(p, "main_grad") and p.main_grad is not None:
                    mg = p.main_grad
                    p.grad = mg if mg.dtype == p.dtype else mg.to(p.dtype)

        # loss_dicts is non-empty only on the last pipeline stage (PP=1: always).
        # Each entry is {'lm loss': (loss_sum, token_count)}.
        loss: float = 0.0
        if loss_dicts:
            lm = loss_dicts[0].get("lm loss")
            if lm is not None:
                # Megatron returns (loss_sum, token_count) as a length-2 tensor
                loss = float(lm[0] / lm[1].clamp(min=1)) if lm.numel() > 1 else float(lm[0])
        self.last_loss = loss

        logger.info("forward_backward: done in %.2fs — loss=%.4f", time.perf_counter() - t0, loss)
        return ForwardBackwardResult(
            loss_fn_output_type="cross_entropy_loss",
            # TODO: extract per-token logprobs by running a second forward-only
            # pass — Megatron fuses loss into the forward pass and does not
            # return logits by default.
            loss_fn_outputs=[],
            metrics={"loss": loss},
        )

    def _optim_step_impl(self, details: OptimStepDetails) -> OptimStepResult:
        t0 = time.perf_counter()
        logger.info("optim_step: step=%d …", self.step)
        params = details.adam_params
        optimizer = self.trainer.optimizer

        # Megatron's DistributedOptimizer stores param groups differently from
        # plain AdamW; iterate over whatever groups are exposed.
        for pg in optimizer.param_groups:
            pg["lr"] = params.learning_rate
            if "betas" in pg:
                pg["betas"] = (params.beta1, params.beta2)
            elif "beta1" in pg:
                pg["beta1"] = params.beta1
                pg["beta2"] = params.beta2
            pg["eps"] = params.eps
            pg["weight_decay"] = params.weight_decay

        # Gradient clipping and norm: Megatron's optimizer exposes clip_grad_norm.
        grad_norm: float = 0.0
        if hasattr(optimizer, "clip_grad_norm"):
            max_norm = params.grad_clip_norm if params.grad_clip_norm > 0 else float("inf")
            grad_norm = float(optimizer.clip_grad_norm(max_norm))
        elif params.grad_clip_norm > 0:
            # Fallback: gather parameters from all PP/TP stages and clip manually.
            all_params = [p for m in self.unwrapped_models for p in m.parameters()]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(all_params, params.grad_clip_norm))

        optimizer.step()
        # Clear the .grad copies set in _forward_backward_impl before calling
        # optimizer.zero_grad() so megatron's zero_grad only sees main_grad.
        for m in self.unwrapped_models:
            for p in m.parameters():
                p.grad = None
        optimizer.zero_grad()
        self.step += 1

        lr = float(optimizer.param_groups[0]["lr"])
        logger.info("optim_step: done in %.2fs — step=%d lr=%.2e grad_norm=%.4f",
                    time.perf_counter() - t0, self.step, lr, grad_norm)
        return OptimStepResult(metrics={
            "step": float(self.step),
            "learning_rate": lr,
            "lr": lr,
            "grad_norm": grad_norm,
        })

    def _to_inference_impl(self) -> ToInferenceResult:
        t0 = time.perf_counter()
        logger.info("to_inference: switching model to eval mode …")
        for m in self.unwrapped_models:
            m.eval()

        # Weight sync: bridge.export_weights() is a COLLECTIVE operation — all TP
        # ranks must call it together.  Only rank 0 actually pushes to vLLM.
        logger.info("to_inference: exporting Megatron weights via bridge …")
        self._ensure_communicator_ready()

        weight_bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_size_bytes = int(os.environ.get("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", "512")) * 1024 * 1024
        current_bytes = 0

        for name, param in self.trainer.bridge.export_weights(self.unwrapped_models):
            if is_rank_zero():
                weight_bucket.append((name, param))
                current_bytes += param.numel() * param.element_size()
                if current_bytes >= bucket_size_bytes:
                    self._flush_weight_bucket(weight_bucket)
                    weight_bucket = []
                    current_bytes = 0

        if is_rank_zero() and weight_bucket:
            self._flush_weight_bucket(weight_bucket)

        if is_rank_zero() and self.vllm_client is not None:
            self.vllm_client.reset_prefix_cache()

        self.mode = "inference"
        logger.info("to_inference: done in %.2fs", time.perf_counter() - t0)
        return ToInferenceResult(mode=self.mode)

    def _flush_weight_bucket(self, bucket: List[Tuple[str, torch.Tensor]]) -> None:
        """Send a bucket of (name, param) pairs to the vLLM rollout server."""
        assert self.vllm_client is not None
        fb = FlattenedTensorBucket(named_tensors=bucket)
        self.vllm_client.update_flattened_params(fb.get_metadata(), fb.get_flattened_tensor())

    def _to_training_impl(self) -> None:
        logger.info("to_training: switching model to train mode …")
        for m in self.unwrapped_models:
            m.train()
        self.mode = "training"

    def _save_state_impl(self, path: str) -> SaveStateResult:
        t0 = time.perf_counter()
        logger.info("save_state: saving to %s …", path)
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # bridge.save_weights() gathers TP/PP shards and writes HF-format safetensors.
        # This is a collective operation (all ranks must call it).
        logger.info("save_state: exporting HF-format checkpoint …")
        self.trainer.bridge.save_weights(
            self.unwrapped_models,
            output_dir=str(ckpt_dir),
            processor=self.processor,
        )

        if is_rank_zero():
            trainer_state = {
                "step": self.step,
                "mode": self.mode,
                "last_loss": self.last_loss,
                "config": self.config.model_dump(),
            }
            torch.save(trainer_state, ckpt_dir / "trainer_state.pt")

        logger.info("save_state: done in %.2fs", time.perf_counter() - t0)
        return SaveStateResult(mode=self.mode)

    def _sample_impl(self, details: SampleDetails) -> SampleResult:
        from swift.infer_engine import RequestConfig
        from swift.infer_engine.protocol import RolloutInferRequest

        t0 = time.perf_counter()
        prompt_tokens = details.prompt.to_ints()
        params = details.sampling_params
        max_tokens = params.max_tokens or 128
        effective_max_tokens = max(1, min(max_tokens, self._rollout_max_model_len - len(prompt_tokens)))

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")

        sequences = []
        for _ in range(details.num_samples):
            rollout_outputs = self.vllm_client.infer(
                [RolloutInferRequest(messages=[{"role": "user", "content": prompt_text}])],
                request_config=RequestConfig(
                    max_tokens=effective_max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                ),
                use_tqdm=False,
            )
            if not rollout_outputs:
                raise RuntimeError("No outputs returned from rollout server.")
            output = rollout_outputs[0]
            response = output.response if hasattr(output, "response") else output
            generated_text = response.choices[0].message.content
            generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            sequences.append(SampledSequence(
                tokens=generated_tokens,
                logprobs=None,
                stop_reason=response.choices[0].finish_reason or "length",
            ))

        logger.info("sample: done in %.2fs — %d sequence(s)", time.perf_counter() - t0, len(sequences))
        return SampleResult(sequences=sequences)

    # ── Status & mode helpers ────────────────────────────────────────────────

    def _ensure_training_mode(self, caller: str) -> None:
        if self.mode == "training":
            return
        logger.info("%s: auto-switching to training mode …", caller)
        self._to_training_impl()

    def get_status(self) -> StatusResult:
        with self._lock:
            # Report the device of the first parameter of the first model stage.
            device = str(next(self.unwrapped_models[0].parameters()).device)
            gpu_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f"cuda:{i}"] = int(torch.cuda.memory_allocated(i))
            return StatusResult(
                mode=self.mode,
                step=self.step,
                model_id=self.config.model_id,
                device=device,
                last_loss=self.last_loss,
                grad_norm=None,
                gpu_memory=gpu_memory,
            )

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.vllm_client is not None:
                try:
                    self.vllm_client.close_communicator()
                except Exception as e:
                    logger.warning("Failed to close VLLMClient: %s", e)
                finally:
                    try:
                        atexit.unregister(self.vllm_client.close_communicator)
                    except Exception:
                        pass
            if self._rollout_process is not None:
                self._kill_rollout_process_tree()

    def _kill_rollout_process_tree(self) -> None:
        if self._rollout_process is None or not self._rollout_process.is_alive():
            return
        pid = self._rollout_process.pid
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            try:
                self._rollout_process.terminate()
            except Exception:
                pass
        self._rollout_process.join(timeout=10)
        if self._rollout_process.is_alive():
            try:
                os.killpg(pid, signal.SIGKILL)
            except Exception:
                try:
                    self._rollout_process.kill()
                except Exception:
                    pass
            self._rollout_process.join(timeout=5)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
