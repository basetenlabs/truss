import logging
import time

import torch
from async_batcher.batcher import AsyncBatcher
from prometheus_client import Gauge, Histogram
from whisper_utils.constants import PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS

logger = logging.getLogger(__name__)


class DiarizerBatchProcessor(AsyncBatcher):
    def __init__(self, segmentation_model, embedding_model, **kwargs):
        super().__init__(**kwargs)

        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.metrics = {
            "diarizer_async_batcher_gauge": Gauge(
                "diarizer_async_batcher_gauge",
                "Gauge of async batcher size",
            ),
            "diarizer_latency_histogram": Histogram(
                "diarizer_latency_histogram",
                "Histogram of diarizer latencies in seconds",
                buckets=PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS,
            ),
        }

    async def process_batch(self, batch):
        logger.info(f"[Async Batcher] Processing batch of size: {len(batch)}")
        self.metrics["diarizer_async_batcher_gauge"].set(len(batch))

        start_time = time.time()

        # Handle optimized case: if batch contains a single tensor (from optimized chainlet)
        # AsyncBatcher wraps single items in a list, so batch is [tensor]
        if len(batch) == 1 and isinstance(batch[0], torch.Tensor):
            aggregated_batch = batch[0]
            batch_sizes = [batch[0].shape[0]]
        else:
            # Stack multiple tensors efficiently - ensure all are on same device
            # Get device from first tensor, or default to CPU
            first_item = batch[0]
            if isinstance(first_item, torch.Tensor):
                device = first_item.device
                aggregated_batch = torch.vstack(
                    [b.to(device) for b in batch if isinstance(b, torch.Tensor)]
                )
                batch_sizes = [b.shape[0] for b in batch if isinstance(b, torch.Tensor)]
            else:
                # Fallback: convert to tensors
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tensor_list = [
                    torch.tensor(b).to(device) if not isinstance(b, torch.Tensor) else b.to(device)
                    for b in batch
                ]
                aggregated_batch = torch.vstack(tensor_list)
                batch_sizes = [b.shape[0] if isinstance(b, torch.Tensor) else len(b) for b in batch]

        # Process models (they handle device placement internally)
        segmentations = self.segmentation_model(aggregated_batch)
        embeddings = self.embedding_model(aggregated_batch, segmentations)

        # Vectorized ungrouping: use torch.split for efficient tensor slicing
        # Keep tensors on GPU until needed
        if isinstance(segmentations, torch.Tensor) and isinstance(embeddings, torch.Tensor):
            # Use torch.split for efficient splitting without copying
            ungrouped_segmentations = torch.split(segmentations, batch_sizes, dim=0)
            ungrouped_embeddings = torch.split(embeddings, batch_sizes, dim=0)
            result = list(zip(ungrouped_segmentations, ungrouped_embeddings))
        else:
            # Fallback to original method if not tensors
            ungrouped_segmentations = []
            ungrouped_embeddings = []
            offset = 0
            for batch_size in batch_sizes:
                ungrouped_segmentations.append(segmentations[offset : offset + batch_size])
                ungrouped_embeddings.append(embeddings[offset : offset + batch_size])
                offset += batch_size
            result = list(zip(ungrouped_segmentations, ungrouped_embeddings))

        self.metrics["diarizer_latency_histogram"].observe(time.time() - start_time)

        return result
