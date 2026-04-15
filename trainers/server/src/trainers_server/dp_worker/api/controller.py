import atexit
import inspect
import logging
import multiprocessing
import os
import signal
import socket
import time
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from swift.arguments import RolloutArguments
from swift.infer_engine import RequestConfig
from swift.infer_engine.protocol import RolloutInferRequest
from swift.model.register import get_model_processor
from swift.pipelines.infer.rollout import rollout_main
from swift.rlhf_trainers.utils import FlattenedTensorBucket
from swift.rlhf_trainers.vllm_client import VLLMClient
from swift.template.register import get_template
from swift.tuners import LoRAConfig, Swift
from torch.optim import AdamW

from trainers_server.shared.models import (
    Datum,
    ForwardBackwardDetails,
    ForwardBackwardResult,
    LoadStateDetails,
    LoadStateResult,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_id) for gpu_id in config.inference.gpus
    )
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
    """Controller that exposes discrete RL training/inference transitions."""

    def __init__(
        self, config: RLControllerConfig, *, model=None, processor=None, template=None
    ) -> None:
        init_t0 = time.perf_counter()
        logger.info("RLController.__init__ starting for model=%s", config.model_id)
        self.config = config
        self._lock = RLock()
        self.mode = "training"
        self.step = 0
        self.last_loss: Optional[float] = None
        self._closed = False
        self._rollout_process: Optional[multiprocessing.Process] = None
        self.vllm_client: Optional[VLLMClient] = None
        self._communicator_ready = False
        self._rollout_port = self._find_free_port()
        self._rollout_group_port = self._find_free_port()
        self._rollout_max_model_len = max(self.config.training.max_length, 4096)

        if not self.config.training.gpus:
            raise ValueError("training.gpus must contain at least one GPU id.")
        if not self.config.inference.gpus:
            raise ValueError("inference.gpus must contain at least one GPU id.")

        training_device = self._training_device()
        load_checkpoint_dir = os.environ.get("BT_LOAD_CHECKPOINT_DIR")

        if model is None or processor is None:
            logger.info(
                "Loading model and processor from %s (dtype=bfloat16, training_device=%s) ...",
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

            if config.lora_rank > 0:
                t0 = time.perf_counter()
                if load_checkpoint_dir:
                    logger.info(
                        "Loading LoRA adapters (rank=%d) from checkpoint %s ...",
                        config.lora_rank,
                        load_checkpoint_dir,
                    )
                    model = Swift.from_pretrained(
                        model, model_id=load_checkpoint_dir, is_trainable=True
                    )
                else:
                    logger.info(
                        "Applying fresh LoRA adapters (rank=%d) ...", config.lora_rank
                    )
                    model = Swift.prepare_model(model, LoRAConfig(r=config.lora_rank))
                logger.info("LoRA adapters ready in %.1fs", time.perf_counter() - t0)
            elif load_checkpoint_dir:
                logger.info(
                    "Overwriting weights from checkpoint %s ...", load_checkpoint_dir
                )
                t0 = time.perf_counter()
                model_cls = type(model)
                del model
                torch.cuda.empty_cache()
                model = model_cls.from_pretrained(
                    load_checkpoint_dir,
                    torch_dtype=torch.bfloat16,
                    device_map={"": training_device},
                )
                logger.info(
                    "Checkpoint weights loaded in %.1fs", time.perf_counter() - t0
                )

        self.model = model
        self.processor = processor

        if template is None:
            logger.info(
                "Building template (max_length=%d) ...", config.training.max_length
            )
            template = get_template(
                processor,
                max_length=config.training.max_length,
                truncation_strategy="left",
            )
        self.template = template
        self.template.set_mode("train")

        logger.info(
            "Initializing AdamW optimizer (params set per optim_step via AdamParams) ..."
        )
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=0.0, weight_decay=0.0)
        self.optimizer.zero_grad(set_to_none=True)

        if load_checkpoint_dir:
            trainer_state_path = Path(load_checkpoint_dir) / "trainer_state.pt"
            if trainer_state_path.exists():
                logger.info("Restoring trainer state from %s ...", trainer_state_path)
                trainer_state = torch.load(
                    trainer_state_path, map_location="cpu", weights_only=False
                )
                self.step = trainer_state.get("step", 0)
                self.last_loss = trainer_state.get("last_loss")
                self.optimizer.load_state_dict(trainer_state["optimizer"])
                logger.info("Trainer state restored — step=%d", self.step)
            else:
                logger.info(
                    "No trainer_state.pt found in %s, starting from step 0.",
                    load_checkpoint_dir,
                )

        logger.info(
            "Launching rollout server on inference GPUs=%s ...", config.inference.gpus
        )
        self._launch_rollout_server()
        logger.info(
            "Connecting VLLMClient to rollout server at 127.0.0.1:%d ...",
            self._rollout_port,
        )
        self._init_vllm_client()
        logger.info(
            "Skipping initial weight sync at startup (rollout loads model weights directly)."
        )
        logger.info(
            "RLController.__init__ complete in %.1fs — mode=%s",
            time.perf_counter() - init_t0,
            self.mode,
        )

    def _training_device(self) -> str:
        return f"cuda:{self.config.training.gpus[0]}"

    def _launch_rollout_server(self) -> None:
        mp_ctx = multiprocessing.get_context("spawn")
        config_data = self.config.model_dump()
        process = mp_ctx.Process(
            target=_rollout_server_entry, args=(config_data, self._rollout_port)
        )
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
            if (
                self._rollout_process is not None
                and not self._rollout_process.is_alive()
            ):
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
                    health_url, timeout=float(_ROLLOUT_RETRY_INTERVAL_SECONDS)
                )
                if health_response.status_code != 200:
                    raise RuntimeError(
                        f"Health endpoint returned status={health_response.status_code}"
                    )

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

    def _ensure_communicator_ready(self) -> None:
        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")
        if self._communicator_ready:
            return
        training_device = self._training_device()
        logger.info(
            "Initializing rollout communicator on training device %s ...",
            training_device,
        )
        try:
            self.vllm_client.init_communicator(device=training_device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize rollout communicator: {exc}"
            ) from exc
        self._communicator_ready = True

    def _iter_weight_buckets(self) -> List[List[Tuple[str, torch.Tensor]]]:
        bucket_size_mb = int(os.environ.get("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", "512"))
        max_bucket_bytes = bucket_size_mb * 1024 * 1024
        named_params = [
            (name, param.detach()) for name, param in self.model.named_parameters()
        ]

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
        return buckets

    def _sync_weights_to_rollout(self) -> None:
        if self.vllm_client is None:
            raise RuntimeError("VLLMClient is not initialized.")
        if not self._communicator_ready:
            raise RuntimeError("Weight sync communicator is not initialized.")

        buckets = self._iter_weight_buckets()
        logger.info(
            "Syncing %d parameter bucket(s) to rollout server ...", len(buckets)
        )
        for i, bucket in enumerate(buckets):
            flat_bucket = FlattenedTensorBucket(named_tensors=bucket)
            self.vllm_client.update_flattened_params(
                flat_bucket.get_metadata(), flat_bucket.get_flattened_tensor()
            )
            logger.info("Weight bucket %d/%d synced", i + 1, len(buckets))
        self.vllm_client.reset_prefix_cache()

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

    def forward_backward(
        self, details: ForwardBackwardDetails
    ) -> ForwardBackwardResult:
        if not details.data:
            raise ValueError("`data` must contain at least one request.")
        with self._lock:
            self._ensure_training_mode("forward_backward")
            if details.loss_fn != "cross_entropy":
                raise ValueError(
                    f"Unsupported loss_fn: {details.loss_fn}. Only 'cross_entropy' is supported."
                )
            t0 = time.perf_counter()
            logger.info(
                "forward_backward: preparing %d pre-tokenized sample(s) ...",
                len(details.data),
            )
            self.model.train()

            batch_tokens: List[List[int]] = []
            reward_values = []
            for datum in details.data:
                tokens = datum.model_input.to_ints()
                if len(tokens) < 2:
                    raise ValueError(
                        "Each datum.model_input must contain at least 2 tokens."
                    )
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
                "input_ids": torch.tensor(
                    input_ids_rows, dtype=torch.long, device=device
                ),
                "attention_mask": torch.tensor(
                    attention_rows, dtype=torch.long, device=device
                ),
            }
            labels = torch.tensor(label_rows, dtype=torch.long, device=device)
            model_forward_params = set(
                inspect.signature(self.model.forward).parameters.keys()
            )
            model_inputs = {
                k: v for k, v in inputs.items() if k in model_forward_params
            }

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
                # Gather logprobs for the actual target tokens.
                # Replace -100 with 0 for gather (masked positions).
                gather_labels = shift_labels.clone()
                gather_labels[gather_labels == -100] = 0
                token_logprobs = log_probs.gather(
                    -1, gather_labels.unsqueeze(-1)
                ).squeeze(-1)

            valid_mask = shift_labels.ne(-100)
            token_loss = token_loss * valid_mask
            valid_counts = valid_mask.sum(dim=-1).clamp(min=1)
            per_sample_loss = token_loss.sum(dim=-1) / valid_counts

            rewards = torch.tensor(
                reward_values,
                dtype=per_sample_loss.dtype,
                device=per_sample_loss.device,
            )
            loss = (per_sample_loss * rewards).mean()

            logger.info("forward_backward: running backward pass ...")
            loss.backward()
            self.last_loss = float(loss.detach().cpu())

            # Build per-sample logprobs output (masked positions excluded).
            loss_fn_outputs = []
            for i in range(token_logprobs.shape[0]):
                mask_i = valid_mask[i]
                lp = token_logprobs[i][mask_i].detach().cpu().tolist()
                loss_fn_outputs.append(
                    {"logprobs": {"data": lp, "dtype": "float32", "shape": [len(lp)]}}
                )

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
            logger.info(
                "optim_step: stepping optimizer (current step=%d) ...", self.step
            )

            # Apply adam_params if provided.
            params = details.adam_params
            for pg in self.optimizer.param_groups:
                pg["lr"] = params.learning_rate
                pg["betas"] = (params.beta1, params.beta2)
                pg["eps"] = params.eps
                pg["weight_decay"] = params.weight_decay

            # Compute gradient norm before clipping.
            grad_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(p.grad.detach().float().pow(2).sum().item())
            grad_norm = grad_norm_sq**0.5

            if params.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), params.grad_clip_norm
                )

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
            return OptimStepResult(
                metrics={
                    "step": float(self.step),
                    "learning_rate": learning_rate,
                    "lr": learning_rate,
                    "grad_norm": grad_norm,
                }
            )

    def to_inference(self) -> ToInferenceResult:
        with self._lock:
            t0 = time.perf_counter()
            logger.info("to_inference: switching from training to inference mode ...")
            self.model.eval()
            logger.info("to_inference: syncing weights to rollout inference server ...")
            self._ensure_communicator_ready()
            self._sync_weights_to_rollout()
            self.mode = "inference"
            logger.info(
                "to_inference: done in %.2fs — mode=%s",
                time.perf_counter() - t0,
                self.mode,
            )
            return ToInferenceResult(mode=self.mode)

    def to_training(self) -> StatusResult:
        with self._lock:
            t0 = time.perf_counter()
            logger.info("to_training: switching from inference to training mode ...")
            self.model.train()
            self.mode = "training"
            logger.info(
                "to_training: done in %.2fs — mode=%s",
                time.perf_counter() - t0,
                self.mode,
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
                len(prompt_tokens),
                details.num_samples,
                effective_max_tokens,
            )

            tokenizer = getattr(self.processor, "tokenizer", self.processor)

            # Decode tokens to text, send through ms-swift rollout, re-tokenize output.
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

            if self.vllm_client is None:
                raise RuntimeError("VLLMClient is not initialized.")

            self.template.set_mode("vllm")
            sequences: List[SampledSequence] = []

            for _ in range(details.num_samples):
                rollout_request = RolloutInferRequest(
                    messages=[{"role": "user", "content": prompt_text}]
                )
                request_config = RequestConfig(
                    max_tokens=effective_max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                )
                rollout_outputs = self.vllm_client.infer(
                    [rollout_request], request_config=request_config, use_tqdm=False
                )
                if not rollout_outputs:
                    raise RuntimeError("No outputs returned from rollout server.")
                output = rollout_outputs[0]
                response = output.response if hasattr(output, "response") else output
                generated_text = response.choices[0].message.content
                generated_tokens = tokenizer.encode(
                    generated_text, add_special_tokens=False
                )
                sequences.append(
                    SampledSequence(
                        tokens=generated_tokens,
                        logprobs=None,  # TODO: extract logprobs from vLLM response
                        stop_reason=response.choices[0].finish_reason or "length",
                    )
                )

            logger.info(
                "sample: done in %.2fs — %d sequence(s)",
                time.perf_counter() - t0,
                len(sequences),
            )
            return SampleResult(sequences=sequences)

    def save_state(self, path: Optional[str] = None) -> SaveStateResult:
        resolved_path = path or os.environ.get("BT_CHECKPOINT_DIR")
        if not resolved_path:
            raise ValueError(
                "save_state requires a path. Pass path= or set the BT_CHECKPOINT_DIR env var."
            )
        with self._lock:
            self._ensure_training_mode("save_state")
            t0 = time.perf_counter()
            logger.info("save_state: saving to %s ...", resolved_path)
            ckpt_dir = Path(resolved_path)
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
            logger.info("save_state: done in %.2fs", time.perf_counter() - t0)
            return SaveStateResult(mode=self.mode)

    def _resolve_load_path(self, path: Optional[str], caller: str) -> Path:
        resolved = path or os.environ.get("BT_LOAD_CHECKPOINT_DIR")
        if not resolved:
            raise ValueError(
                f"{caller} requires a path. Pass path= or set the BT_LOAD_CHECKPOINT_DIR env var."
            )
        ckpt_dir = Path(resolved)
        if not ckpt_dir.exists():
            raise ValueError(f"{caller}: checkpoint directory not found: {ckpt_dir}")
        return ckpt_dir

    def load_state(self, details: LoadStateDetails) -> LoadStateResult:
        ckpt_dir = self._resolve_load_path(details.path, "load_state")
        with self._lock:
            t0 = time.perf_counter()
            logger.info("load_state: loading model weights from %s ...", ckpt_dir)
            model_cls = type(self.model)
            del self.optimizer
            del self.model
            torch.cuda.empty_cache()
            if self.config.lora_rank > 0:
                base_model, _ = get_model_processor(
                    self.config.model_id,
                    torch_dtype=_parse_torch_dtype("bfloat16"),
                    device_map={"": self._training_device()},
                    use_hf=True,
                )
                self.model = Swift.from_pretrained(
                    base_model, model_id=str(ckpt_dir), is_trainable=True
                )
            else:
                self.model = model_cls.from_pretrained(
                    str(ckpt_dir),
                    torch_dtype=torch.bfloat16,
                    device_map={"": self._training_device()},
                )
            self.model.train()
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(trainable_params, lr=0.0, weight_decay=0.0)
            self.optimizer.zero_grad(set_to_none=True)
            self.mode = "training"
            self._communicator_ready = False
            logger.info(
                "load_state: done in %.2fs — weights loaded, optimizer reset",
                time.perf_counter() - t0,
            )
            return LoadStateResult(mode=self.mode, step=self.step)

    def load_state_with_optimizer(self, details: LoadStateDetails) -> LoadStateResult:
        ckpt_dir = self._resolve_load_path(details.path, "load_state_with_optimizer")
        with self._lock:
            t0 = time.perf_counter()
            logger.info("load_state_with_optimizer: loading from %s ...", ckpt_dir)
            model_cls = type(self.model)
            del self.optimizer
            del self.model
            torch.cuda.empty_cache()
            if self.config.lora_rank > 0:
                base_model, _ = get_model_processor(
                    self.config.model_id,
                    torch_dtype=_parse_torch_dtype("bfloat16"),
                    device_map={"": self._training_device()},
                    use_hf=True,
                )
                self.model = Swift.from_pretrained(
                    base_model, model_id=str(ckpt_dir), is_trainable=True
                )
            else:
                self.model = model_cls.from_pretrained(
                    str(ckpt_dir),
                    torch_dtype=torch.bfloat16,
                    device_map={"": self._training_device()},
                )
            self.model.train()
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(trainable_params, lr=0.0, weight_decay=0.0)
            self.optimizer.zero_grad(set_to_none=True)

            trainer_state_path = ckpt_dir / "trainer_state.pt"
            if trainer_state_path.exists():
                trainer_state = torch.load(
                    trainer_state_path, map_location="cpu", weights_only=False
                )
                self.step = trainer_state.get("step", 0)
                self.last_loss = trainer_state.get("last_loss")
                self.optimizer.load_state_dict(trainer_state["optimizer"])
                logger.info("load_state_with_optimizer: restored step=%d", self.step)
            else:
                logger.warning(
                    "load_state_with_optimizer: no trainer_state.pt found, optimizer state not restored"
                )

            self.mode = "training"
            self._communicator_ready = False
            logger.info(
                "load_state_with_optimizer: done in %.2fs", time.perf_counter() - t0
            )
            return LoadStateResult(mode=self.mode, step=self.step)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self.vllm_client is not None:
                try:
                    self.vllm_client.close_communicator()
                except Exception as e:
                    logger.warning("Failed to close VLLM communicator: %s", e)
                finally:
                    # init_communicator() registers an atexit callback that can fire
                    # after test/process teardown when log streams are already closed.
                    # Unregister here to prevent noisy shutdown-time errors.
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

        # Kill the entire rollout process group (uvicorn + worker + EngineCore).
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
                grad_norm_sq += float(
                    torch.sum(param.grad.detach().float() ** 2).item()
                )
            grad_norm = grad_norm_sq**0.5 if has_grad else None
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
