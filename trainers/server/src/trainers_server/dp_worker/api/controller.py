import inspect
import logging
import multiprocessing
import os
import socket
import signal
import time
import atexit
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Sequence, Tuple

import requests
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from swift.arguments import RolloutArguments
from swift.infer_engine import RequestConfig
from swift.infer_engine.protocol import RolloutInferRequest
from swift.model.register import get_model_processor
from swift.pipelines.infer.rollout import rollout_main
from swift.rlhf_trainers.utils import FlattenedTensorBucket
from swift.rlhf_trainers.vllm_client import VLLMClient
from swift.template.register import get_template

from trainers_server.dp_worker.distributed import (
    WorkerOp,
    barrier,
    broadcast_object,
    get_local_rank,
    get_rank,
    is_distributed,
    is_rank_zero,
)
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


def _parse_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype.lower())


_ROLLOUT_STARTUP_TIMEOUT_SECONDS = 300
_ROLLOUT_RETRY_INTERVAL_SECONDS = 5


def _rollout_server_entry(config_data: Dict, rollout_port: int) -> None:
    # Put rollout and all descendants in their own process group so parent can
    # cleanly terminate the whole tree.
    os.setsid()
    # Ensure rollout dies if the parent controller process dies unexpectedly.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        logger.warning("Failed to set parent-death signal for rollout process.")

    config = RLControllerConfig(**config_data)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config.inference.gpus)
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


class RLController:
    """Controller that exposes discrete RL training/inference transitions.

    Multi-GPU (FSDP) support
    ------------------------
    When launched via ``torchrun --nproc_per_node=N``, ``torch.distributed``
    must already be initialized before constructing this class (see
    ``distributed.init_process_group``).  Each rank builds its own
    ``RLController`` instance.

    * **Rank 0** runs the FastAPI server and owns the vLLM rollout process.
      It broadcasts a ``WorkerOp`` to all other ranks before each collective
      operation so they participate in the FSDP forward/backward.

    * **Ranks 1…N-1** call :meth:`worker_loop` which blocks on broadcasts
      and forwards them to the corresponding ``_*_impl`` methods.

    The FSDP-aware weight-sync path gathers the full parameter state on rank 0
    (using :func:`torch.distributed.fsdp.FullStateDictConfig`) before sending
    weights to the vLLM rollout server over NCCL.
    """

    def __init__(
        self,
        config: RLControllerConfig,
        *,
        model=None,
        processor=None,
        template=None,
    ) -> None:
        init_t0 = time.perf_counter()
        logger.info("RLController.__init__ starting for model=%s rank=%d", config.model_id, get_rank())
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
        self._rollout_max_model_len = max(self.config.training.max_length, 4096)

        # Distributed state
        self._is_distributed = is_distributed()
        self._rank = get_rank()
        self._local_rank = get_local_rank()
        self._is_fsdp = False

        if not self.config.training.gpus:
            raise ValueError("training.gpus must contain at least one GPU id.")
        if not self.config.inference.gpus:
            raise ValueError("inference.gpus must contain at least one GPU id.")

        training_device = self._training_device()
        if model is None or processor is None:
            logger.info(
                "Loading model and processor for %s (dtype=bfloat16, device=%s) ...",
                config.model_id,
                training_device,
            )
            t0 = time.perf_counter()
            model, processor = get_model_processor(
                config.model_id,
                torch_dtype=_parse_torch_dtype("bfloat16"),
                device_map={"": training_device},
                use_hf=True,
            )
            logger.info("Model and processor loaded in %.1fs", time.perf_counter() - t0)
        self.model = model
        self.processor = processor

        # Wrap in FSDP when running with multiple ranks.
        if self._is_distributed:
            self.model = self._wrap_fsdp(self.model)
            self._is_fsdp = True
            logger.info("Model wrapped with FSDP (rank=%d)", self._rank)

        if template is None:
            logger.info("Building template (max_length=%d) ...", config.training.max_length)
            template = get_template(
                processor,
                max_length=config.training.max_length,
                truncation_strategy="left",
            )
        self.template = template
        self.template.set_mode("train")

        logger.info("Initializing AdamW optimizer ...")
        self.optimizer = AdamW(self.model.parameters(), lr=0.0, weight_decay=0.0)
        self.optimizer.zero_grad(set_to_none=True)

        # Only rank 0 owns the rollout server and vLLM client.
        if not self._is_distributed or is_rank_zero():
            self._rollout_port = self._find_free_port()
            self._rollout_group_port = self._find_free_port()
            logger.info("Launching rollout server on inference GPUs=%s ...", config.inference.gpus)
            self._launch_rollout_server()
            logger.info("Connecting VLLMClient to rollout server at 127.0.0.1:%d ...", self._rollout_port)
            self._init_vllm_client()
            logger.info("Skipping initial weight sync at startup (rollout loads model weights directly).")

        logger.info(
            "RLController.__init__ complete in %.1fs — mode=%s rank=%d fsdp=%s",
            time.perf_counter() - init_t0,
            self.mode,
            self._rank,
            self._is_fsdp,
        )

    # -----------------------------------------------------------------------
    # Device / FSDP helpers
    # -----------------------------------------------------------------------

    def _training_device(self) -> str:
        """Select the CUDA device for this rank.

        In distributed mode, each rank uses its ``LOCAL_RANK`` as the device
        index, which is the standard torchrun convention.  In single-rank mode,
        falls back to ``config.training.gpus[0]``.
        """
        if self._is_distributed:
            return f"cuda:{self._local_rank}"
        return f"cuda:{self.config.training.gpus[0]}"

    def _wrap_fsdp(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap *model* in FullyShardedDataParallel."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        from functools import partial

        # Shard parameters *and* optimizer states across all ranks.
        sharding_strategy = ShardingStrategy.FULL_SHARD

        # Keep compute in bf16 but accumulate gradients in fp32 for stability.
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

        # Wrap sub-modules that have >= 1 M parameters so FSDP shards at a
        # sensible granularity without splitting tiny layers.
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e6))

        wrapped = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.device(f"cuda:{self._local_rank}"),
        )
        return wrapped

    # -----------------------------------------------------------------------
    # Rollout server lifecycle
    # -----------------------------------------------------------------------

    def _launch_rollout_server(self) -> None:
        mp_ctx = multiprocessing.get_context("spawn")
        config_data = self.config.model_dump()
        process = mp_ctx.Process(target=_rollout_server_entry, args=(config_data, self._rollout_port))
        process.start()
        self._rollout_process = process

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    def _init_vllm_client(self) -> None:
        deadline = time.time() + float(_ROLLOUT_STARTUP_TIMEOUT_SECONDS)
        attempt = 0
        last_error: Optional[Exception] = None

        while time.time() < deadline:
            attempt += 1
            if self._rollout_process is not None and not self._rollout_process.is_alive():
                exit_code = self._rollout_process.exitcode
                raise RuntimeError(
                    f"Rollout server process exited during startup (exit_code={exit_code})."
                )

            logger.info(
                "Waiting for rollout server (attempt %d, port=%d) ...",
                attempt,
                self._rollout_port,
            )
            try:
                health_url = f"http://127.0.0.1:{self._rollout_port}/health/"
                health_response = requests.get(
                    health_url,
                    timeout=float(_ROLLOUT_RETRY_INTERVAL_SECONDS),
                )
                if health_response.status_code != 200:
                    raise RuntimeError(f"Health endpoint returned status={health_response.status_code}")

                client = VLLMClient(
                    hosts=["127.0.0.1"],
                    server_ports=[self._rollout_port],
                    group_ports=[self._rollout_group_port],
                    connection_timeout=float(_ROLLOUT_RETRY_INTERVAL_SECONDS),
                )
                self.vllm_client = client
                logger.info("Connected to rollout server and health endpoint is ready.")
                return
            except Exception as e:
                last_error = e
                logger.info("Rollout server not ready yet: %s", e)
                time.sleep(float(_ROLLOUT_RETRY_INTERVAL_SECONDS))

        raise RuntimeError(
            "Timed out waiting for rollout server startup after "
            f"{_ROLLOUT_STARTUP_TIMEOUT_SECONDS}s. Last error: {last_error}"
        )

    # -----------------------------------------------------------------------
    # Weight sync to rollout (FSDP-aware)
    # -----------------------------------------------------------------------

    def _ensure_communicator_ready(self) -> None:
        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")
        if self._communicator_ready:
            return
        training_device = self._training_device()
        logger.info("Initializing rollout communicator on training device %s ...", training_device)
        try:
            self.vllm_client.init_communicator(device=training_device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize rollout communicator: {exc}") from exc
        self._communicator_ready = True

    def _sync_weights_to_rollout(self) -> None:
        """Sync training model weights to the vLLM rollout server.

        With FSDP, we first gather the full parameter state on rank 0, then
        rank 0 sends the weights via the NCCL communicator.  Other ranks
        participate in the FSDP all-gather but do not interact with vLLM.

        Without FSDP (single-rank), this is identical to the original
        per-bucket streaming approach.
        """
        if self._is_fsdp:
            self._sync_weights_fsdp()
        else:
            self._sync_weights_single_rank()

    def _sync_weights_single_rank(self) -> None:
        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")
        if not self._communicator_ready:
            raise RuntimeError("Weight sync communicator is not initialized.")

        named_params = [(name, param.detach()) for name, param in self.model.named_parameters()]
        self._push_named_params(named_params)

    def _sync_weights_fsdp(self) -> None:
        """Gather full FSDP state on rank 0, then rank 0 pushes to vLLM."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # All ranks participate in the all-gather; only rank 0 gets non-empty dict.
        cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.model.state_dict()

        if not is_rank_zero():
            return

        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized (rank 0).")
        if not self._communicator_ready:
            raise RuntimeError("Weight sync communicator is not initialized.")

        named_params = list(state_dict.items())
        self._push_named_params(named_params)

    def _push_named_params(self, named_params: List[Tuple[str, torch.Tensor]]) -> None:
        """Stream named parameters to the vLLM rollout server in buckets."""
        assert self.vllm_client is not None
        bucket_size_mb = int(os.environ.get("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", "512"))
        max_bucket_bytes = bucket_size_mb * 1024 * 1024

        buckets: List[List[Tuple[str, torch.Tensor]]] = []
        current_bucket: List[Tuple[str, torch.Tensor]] = []
        current_bytes = 0
        for name, param in named_params:
            param_bytes = int(param.numel() * param.element_size())
            if current_bucket and current_bytes + param_bytes > max_bucket_bytes:
                buckets.append(current_bucket)
                current_bucket = []
                current_bytes = 0
            current_bucket.append((name, param))
            current_bytes += param_bytes
        if current_bucket:
            buckets.append(current_bucket)

        logger.info("Syncing %d parameter bucket(s) to rollout server ...", len(buckets))
        for i, bucket in enumerate(buckets):
            flat_bucket = FlattenedTensorBucket(named_tensors=bucket)
            self.vllm_client.update_flattened_params(flat_bucket.get_metadata(), flat_bucket.get_flattened_tensor())
            logger.info("Weight bucket %d/%d synced", i + 1, len(buckets))
        self.vllm_client.reset_prefix_cache()

    # -----------------------------------------------------------------------
    # Misc helpers
    # -----------------------------------------------------------------------

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

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

    # -----------------------------------------------------------------------
    # Broadcast helpers (used by public methods on rank 0)
    # -----------------------------------------------------------------------

    def _broadcast_op(self, op: WorkerOp, payload=None) -> None:
        """Rank 0: broadcast *op* + *payload* to all worker ranks."""
        if self._is_distributed:
            broadcast_object((op, payload), src=0)

    # -----------------------------------------------------------------------
    # Public API — called by server.py (rank 0 only)
    # -----------------------------------------------------------------------

    def forward_backward(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
        if not details.data:
            raise ValueError("`data` must contain at least one request.")
        with self._lock:
            self._ensure_training_mode("forward_backward")
            if details.loss_fn != "cross_entropy":
                raise ValueError(f"Unsupported loss_fn: {details.loss_fn}. Only 'cross_entropy' is supported.")
            self._broadcast_op(WorkerOp.FORWARD_BACKWARD, details.model_dump())
            return self._forward_backward_impl(details)

    def optim_step(self, details: OptimStepDetails) -> OptimStepResult:
        with self._lock:
            self._ensure_training_mode("optim_step")
            self._broadcast_op(WorkerOp.OPTIM_STEP, details.model_dump())
            return self._optim_step_impl(details)

    def to_inference(self) -> ToInferenceResult:
        with self._lock:
            self._broadcast_op(WorkerOp.TO_INFERENCE)
            return self._to_inference_impl()

    def to_training(self) -> StatusResult:
        with self._lock:
            self._broadcast_op(WorkerOp.TO_TRAINING)
            self._to_training_impl()
            return self.get_status()

    def sample(self, details: SampleDetails) -> SampleResult:
        with self._lock:
            if self.mode != "inference":
                raise RuntimeError("sample is only allowed in inference mode.")
            # Sampling is purely rank-0: only rank 0 talks to the vLLM server.
            return self._sample_impl(details)

    def save_state(self, path: str) -> SaveStateResult:
        with self._lock:
            self._ensure_training_mode("save_state")
            self._broadcast_op(WorkerOp.SAVE_STATE, path)
            return self._save_state_impl(path)

    # -----------------------------------------------------------------------
    # Implementation methods — called on ALL ranks
    # -----------------------------------------------------------------------

    def _forward_backward_impl(self, details: ForwardBackwardDetails) -> ForwardBackwardResult:
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

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

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
        inputs = {
            "input_ids": torch.tensor(input_ids_rows, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        }
        labels = torch.tensor(label_rows, dtype=torch.long, device=device)
        model_forward_params = set(inspect.signature(self.model.forward).parameters.keys())
        model_inputs = {k: v for k, v in inputs.items() if k in model_forward_params}

        logger.info("forward_backward: running forward pass ...")
        outputs = self.model(**model_inputs)
        logits = outputs.logits

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
            time.perf_counter() - t0, self.last_loss, len(details.data),
        )
        return ForwardBackwardResult(
            loss_fn_output_type="per_token_logprobs",
            loss_fn_outputs=loss_fn_outputs,
            metrics={"loss": self.last_loss},
        )

    def _optim_step_impl(self, details: OptimStepDetails) -> OptimStepResult:
        t0 = time.perf_counter()
        logger.info("optim_step: stepping optimizer (current step=%d) ...", self.step)

        params = details.adam_params
        for pg in self.optimizer.param_groups:
            pg["lr"] = params.learning_rate
            pg["betas"] = (params.beta1, params.beta2)
            pg["eps"] = params.eps
            pg["weight_decay"] = params.weight_decay

        # FSDP provides clip_grad_norm_ which handles the distributed gradient
        # norm reduction internally; fall back to the standard PyTorch call for
        # single-rank mode.
        if self._is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            max_norm = params.grad_clip_norm if params.grad_clip_norm > 0 else float("inf")
            grad_norm = float(FSDP.clip_grad_norm_(self.model, max_norm))
        else:
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

    def _to_inference_impl(self) -> ToInferenceResult:
        t0 = time.perf_counter()
        logger.info("to_inference: switching to inference mode ...")
        self.model.eval()
        if not self._is_distributed or is_rank_zero():
            logger.info("to_inference: syncing weights to rollout inference server ...")
            self._ensure_communicator_ready()
            self._sync_weights_to_rollout()
        elif self._is_distributed:
            # Non-rank-0: still participates in the FSDP all-gather for weight
            # collection, but does not interact with vLLM.
            self._sync_weights_fsdp_non_rank0()
        self.mode = "inference"
        logger.info("to_inference: done in %.2fs", time.perf_counter() - t0)
        return ToInferenceResult(mode=self.mode)

    def _sync_weights_fsdp_non_rank0(self) -> None:
        """Participate in the FSDP all-gather without sending weights to vLLM."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            # Non-rank-0: state_dict is empty (rank0_only=True), but we must
            # call this to participate in the collective.
            self.model.state_dict()

    def _to_training_impl(self) -> None:
        logger.info("to_training: switching to training mode ...")
        self.model.train()
        self.mode = "training"

    def _save_state_impl(self, path: str) -> SaveStateResult:
        t0 = time.perf_counter()
        logger.info("save_state: saving to %s ...", path)

        if self._is_fsdp:
            self._save_state_fsdp(path)
        else:
            self._save_state_single_rank(path)

        logger.info("save_state: done in %.2fs", time.perf_counter() - t0)
        return SaveStateResult(mode=self.mode)

    def _save_state_single_rank(self, path: str) -> None:
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("save_state: saving model ...")
        self.model.save_pretrained(str(ckpt_dir))
        logger.info("save_state: saving processor ...")
        self.processor.save_pretrained(str(ckpt_dir))
        trainer_state = {
            "step": self.step,
            "mode": self.mode,
            "last_loss": self.last_loss,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
        }
        logger.info("save_state: saving trainer state (step=%d) ...", self.step)
        torch.save(trainer_state, ckpt_dir / "trainer_state.pt")

    def _save_state_fsdp(self, path: str) -> None:
        """Gather full model + optimizer state from FSDP shards, then save on rank 0."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullOptimStateDictConfig,
            StateDictType,
            OptimStateDictType,
        )

        # Gather full model parameters onto rank 0.
        model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, model_cfg):
            model_state = self.model.state_dict()

        # Gather full optimizer state onto rank 0.
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT,
                                   model_cfg, optim_cfg):
            optim_state = FSDP.optim_state_dict(self.model, self.optimizer)

        if not is_rank_zero():
            return

        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("save_state: saving model weights (rank 0) ...")
        torch.save(model_state, ckpt_dir / "model_state.pt")
        logger.info("save_state: saving processor ...")
        self.processor.save_pretrained(str(ckpt_dir))
        trainer_state = {
            "step": self.step,
            "mode": self.mode,
            "last_loss": self.last_loss,
            "optimizer": optim_state,
            "config": self.config.model_dump(),
        }
        logger.info("save_state: saving trainer state (step=%d) ...", self.step)
        torch.save(trainer_state, ckpt_dir / "trainer_state.pt")

    # -----------------------------------------------------------------------
    # Sampling (rank 0 only — talks to vLLM)
    # -----------------------------------------------------------------------

    def _sample_impl(self, details: SampleDetails) -> SampleResult:
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

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")

        self.template.set_mode("vllm")
        sequences: List[SampledSequence] = []

        for _ in range(details.num_samples):
            rollout_request = RolloutInferRequest(
                messages=[{"role": "user", "content": prompt_text}],
            )
            request_config = RequestConfig(
                max_tokens=effective_max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
            )
            rollout_outputs = self.vllm_client.infer(
                [rollout_request],
                request_config=request_config,
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
                logprobs=None,  # TODO: extract logprobs from vLLM response
                stop_reason=response.choices[0].finish_reason or "length",
            ))

        logger.info("sample: done in %.2fs — %d sequence(s)", time.perf_counter() - t0, len(sequences))
        return SampleResult(sequences=sequences)

    # -----------------------------------------------------------------------
    # Worker loop (called on ranks 1..N-1 in main.py)
    # -----------------------------------------------------------------------

    def worker_loop(self) -> None:
        """Blocking loop for non-rank-0 worker processes.

        Receives broadcasted ``WorkerOp`` codes from rank 0 and executes the
        corresponding collective operation so that FSDP can participate.
        Returns when a ``SHUTDOWN`` op is received.
        """
        logger.info("Worker rank=%d entering worker_loop ...", self._rank)
        while True:
            payload = broadcast_object(None, src=0)
            op, data = payload
            op = WorkerOp(op)

            if op == WorkerOp.SHUTDOWN:
                logger.info("Worker rank=%d received SHUTDOWN", self._rank)
                break
            elif op == WorkerOp.FORWARD_BACKWARD:
                details = ForwardBackwardDetails.model_validate(data)
                self._forward_backward_impl(details)
            elif op == WorkerOp.OPTIM_STEP:
                details = OptimStepDetails.model_validate(data)
                self._optim_step_impl(details)
            elif op == WorkerOp.TO_INFERENCE:
                self._to_inference_impl()
            elif op == WorkerOp.TO_TRAINING:
                self._to_training_impl()
            elif op == WorkerOp.SAVE_STATE:
                assert isinstance(data, str)
                self._save_state_impl(data)
            else:
                logger.warning("Worker rank=%d received unknown op=%s, skipping.", self._rank, op)

        logger.info("Worker rank=%d exited worker_loop.", self._rank)

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def get_status(self) -> StatusResult:
        with self._lock:
            device = str(self._model_device())
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

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._is_distributed and is_rank_zero():
                # Signal worker ranks to exit their worker_loop.
                try:
                    broadcast_object((WorkerOp.SHUTDOWN, None), src=0)
                except Exception as e:
                    logger.warning("Failed to broadcast SHUTDOWN to workers: %s", e)

            if self.vllm_client is not None:
                try:
                    self.vllm_client.close_communicator()
                except Exception as e:
                    logger.warning("Failed to close VLLM communicator: %s", e)
                finally:
                    try:
                        atexit.unregister(self.vllm_client.close_communicator)
                    except Exception:
                        pass

            if self._rollout_process is not None:
                self._kill_rollout_process_tree()

    def _kill_rollout_process_tree(self) -> None:
        if self._rollout_process is None:
            return
        if not self._rollout_process.is_alive():
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
            try:
                self._rollout_process.terminate()
            except Exception:
                pass

        self._rollout_process.join(timeout=10)
        if self._rollout_process.is_alive():
            logger.warning("Rollout process group did not exit in time, force killing.")
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception as e:
                logger.warning("Failed to SIGKILL rollout process group: %s", e)
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
