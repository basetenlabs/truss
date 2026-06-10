from prometheus_client import Counter, Gauge, Histogram
from whisper_utils.constants import (
    HEALTH_CHECK_LATENCY_BUCKETS_SECONDS,
    PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS,
)


def create_streaming_metrics() -> dict[str, Counter | Gauge | Histogram]:
    return {
        "partial_latency_histogram": Histogram(
            "partial_latency_histogram",
            "Histogram of partial latencies in seconds  (interval between consecutive partial sends)",
            buckets=PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS,
        ),
        "final_latency_histogram": Histogram(
            "final_latency_histogram",
            "Histogram of final latencies in seconds (interval between consecutive final sends)",
            ["diarized"],
            buckets=PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS,
        ),
        "transcription_latency_histogram": Histogram(
            "transcription_latency_histogram",
            "Histogram of transcription latencies in seconds",
        ),
        "assignment_queue_gauge": Gauge(
            "assignment_queue_gauge",
            "Gauge of assignment queue size",
        ),
        "diarization_queue_gauge": Gauge(
            "diarization_queue_gauge",
            "Gauge of diarization queue size",
        ),
        "transcription_queue_gauge": Gauge(
            "transcription_queue_gauge",
            "Gauge of transcription queue size",
        ),
        "diarization_timeout_counter": Counter(
            "diarization_timeout_counter",
            "Counter of diarization timeout",
        ),
        "num_active_connections_gauge": Gauge(
            "num_active_connections_gauge",
            "Gauge of number of active connections",
        ),
        "num_dropped_partial_transcripts_counter": Counter(
            "num_dropped_partial_transcripts_counter",
            "Counter of number of dropped partial transcripts",
        ),
        "pre_assignment_latency_histogram": Histogram(
            "pre_assignment_latency_histogram",
            "Histogram of pre-assignment latencies in seconds",
            buckets=PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS,
        ),
        "audio_processed_seconds_counter": Counter(
            "audio_processed_seconds_counter",
            "Counter of audio processed seconds",
        ),
        "health_check_latency_seconds_histogram": Histogram(
            "health_check_latency_seconds_histogram",
            "Histogram of health check latency in seconds",
            buckets=HEALTH_CHECK_LATENCY_BUCKETS_SECONDS,
        ),
        "pipeline_partial_latency_seconds_histogram": Histogram(
            "pipeline_partial_latency_seconds_histogram",
            "Histogram of pipeline latency for partial transcripts in seconds (from _handle_audio_chunk to send_json_safe)",
            buckets=HEALTH_CHECK_LATENCY_BUCKETS_SECONDS,
        ),
        "pipeline_final_latency_seconds_histogram": Histogram(
            "pipeline_final_latency_seconds_histogram",
            "Histogram of pipeline latency for final transcripts in seconds (from _handle_audio_chunk to send_json_safe)",
            ["diarized"],
            buckets=HEALTH_CHECK_LATENCY_BUCKETS_SECONDS,
        )
        # Memory leak monitoring metrics
        # "final_segments_buffer_size_gauge": Gauge(
        #     "final_segments_buffer_size_gauge",
        #     "Gauge of final segments buffer size per stream",
        #     ["stream_id"],
        # ),
        # "accumulated_diarization_size_gauge": Gauge(
        #     "accumulated_diarization_size_gauge",
        #     "Gauge of accumulated diarization segments count per stream",
        #     ["stream_id"],
        # ),
        # "pending_assignments_size_gauge": Gauge(
        #     "pending_assignments_size_gauge",
        #     "Gauge of pending assignments list size per stream",
        #     ["stream_id"],
        # ),
        # "stream_state_memory_usage_gauge": Gauge(
        #     "stream_state_memory_usage_gauge",
        #     "Gauge of estimated StreamState memory usage in bytes per stream",
        #     ["stream_id"],
        # ),
    }
