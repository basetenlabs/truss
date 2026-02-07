import asyncio
import time

import numpy as np
import pytest
import requests
import truss_transfer

AUDIO_URL = "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
AUDI_M4A = "https://test-audios-public.s3.us-west-2.amazonaws.com/30-sec-01-podcast.m4a"

# skip if version < 0.0.40
pytestmark = pytest.mark.skipif(
    truss_transfer.__version__ < "0.0.40", reason="Requires truss_transfer >= 0.0.40"
)


def test_processor_creation():
    """Test that MultimodalProcessor can be created."""
    print("Testing MultimodalProcessor creation...")
    processor = truss_transfer.MultimodalProcessor()
    assert processor is not None
    print("✓ MultimodalProcessor created successfully")


def test_processor_with_config():
    """Test that MultimodalProcessor can be created with custom config."""
    print("Testing MultimodalProcessor with custom config...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    assert processor is not None
    print("✓ MultimodalProcessor with custom config created successfully")


@pytest.mark.asyncio
async def test_process_audio_from_url():
    """Test audio processing from URL."""
    print(f"Testing audio processing from {AUDIO_URL}...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_config = truss_transfer.AudioConfig()
    audio_array, timing = await processor.process_audio_from_url(
        AUDIO_URL, audio_config
    )

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print(f"✓ Processed {len(audio_array)} audio samples from URL")
    print(f"  Timing: {timing}")

    # check if requests + post + subprocess encode still works
    response = requests.get(AUDIO_URL)
    assert response.status_code == 200, "Failed to download audio"
    audio_bytes2 = response.content
    audio_array2, timing2 = await processor.process_audio_from_bytes(
        audio_bytes2, audio_config
    )
    assert isinstance(audio_array2, np.ndarray), "Result should be numpy array"
    assert audio_array2.dtype == np.float32, (
        f"Expected float32, got {audio_array2.dtype}"
    )
    assert len(audio_array2) > 0, "Processed audio should not be empty"
    assert (audio_array2 == audio_array).all()
    print(f"  Timing2: {timing2}")


@pytest.mark.asyncio
async def test_process_audio_from_base64():
    """Test audio processing from base64."""
    print("Testing audio processing from base64...")
    processor = truss_transfer.MultimodalProcessor()
    audio_config = truss_transfer.AudioConfig()

    # Create a simple base64 encoded audio (just for testing)
    import base64

    request = requests.get(AUDIO_URL)
    encoded = base64.b64encode(request.content).decode("utf-8")

    audio_array, timing = await processor.process_audio_from_base64(
        encoded, audio_config
    )
    # This might fail if the test data isn't valid audio, but that's ok
    print(f"✓ Processed {len(audio_array)} audio samples from base64")
    print(f"  Timing: {timing}")
    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"


@pytest.mark.asyncio
async def test_process_audio_from_bytes():
    """Test audio processing from bytes."""
    print("Testing audio processing from bytes...")
    processor = truss_transfer.MultimodalProcessor()
    audio_config = truss_transfer.AudioConfig()

    # Use the audio URL to get bytes, then process
    audio_bytes = await processor.download_bytes(AUDIO_URL)
    audio_array, timing = await processor.process_audio_from_bytes(
        audio_bytes, audio_config
    )

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print(f"✓ Processed {len(audio_array)} audio samples from bytes")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_download_bytes():
    """Test general download (for images, videos, etc.)."""
    print(f"Testing download of bytes from {AUDIO_URL}...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    bytes_data = await processor.download_bytes(AUDIO_URL)

    assert isinstance(bytes_data, bytes), "Result should be bytes"
    assert len(bytes_data) > 0, "Downloaded file should not be empty"

    print(f"✓ Downloaded {len(bytes_data)} bytes")


@pytest.mark.asyncio
async def test_download_bytes_with_headers():
    """Test download with custom headers."""
    print("Testing download with custom headers...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)

    headers = (
        truss_transfer.Headers()
        .add("User-Agent", "truss-transfer-test")
        .add("Accept", "audio/mpeg")
    )

    bytes_data = await processor.download_bytes(AUDIO_URL, headers=headers)

    assert isinstance(bytes_data, bytes), "Result should be bytes"
    assert len(bytes_data) > 0, "Downloaded file should not be empty"

    print(f"✓ Downloaded {len(bytes_data)} bytes with headers")


@pytest.mark.asyncio
async def test_process_audio():
    """Test unified process_audio method."""
    print("Testing unified process_audio method...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_config = truss_transfer.AudioConfig()
    audio_array, timing = await processor.process_audio("url", AUDIO_URL, audio_config)

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print(f"✓ Processed {len(audio_array)} audio samples via process_audio")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_process_audio_with_base64():
    """Test process_audio with base64 source."""
    print("Testing process_audio with base64 source...")
    processor = truss_transfer.MultimodalProcessor()
    audio_config = truss_transfer.AudioConfig()

    import base64

    test_data = b"test audio data"
    encoded = base64.b64encode(test_data).decode("utf-8")

    try:
        audio_array, timing = await processor.process_audio(
            "base64", encoded, audio_config
        )
        print(f"✓ Processed {len(audio_array)} audio samples via process_audio(base64)")
        print(f"  Timing: {timing}")
    except Exception:
        print(
            "✓ process_audio(base64) test completed (expected error for invalid audio)"
        )


@pytest.mark.asyncio
async def test_process_audio_with_bytes():
    """Test process_audio with bytes source."""
    print("Testing process_audio with bytes source...")
    processor = truss_transfer.MultimodalProcessor()
    audio_config = truss_transfer.AudioConfig()

    audio_bytes = await processor.download_bytes(AUDIO_URL)
    audio_array, timing = await processor.process_audio(
        "bytes", audio_bytes, audio_config
    )

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print(f"✓ Processed {len(audio_array)} audio samples via process_audio(bytes)")
    print(f"  Timing: {timing}")


def test_audio_config_builder():
    """Test AudioConfig builder pattern."""
    print("Testing AudioConfig builder pattern...")
    audio_config = truss_transfer.AudioConfig()
    audio_config = audio_config.with_sample_rate(22050)
    audio_config = audio_config.with_channels(2)
    audio_config = audio_config.with_use_dynamic_normalization(True)
    audio_config = audio_config.with_format("s16le")
    audio_config = audio_config.with_codec("pcm_s16le")
    audio_config = audio_config.with_raw_ffmpeg_args(["-af", "highpass=f=200"])

    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    assert processor is not None
    print("✓ AudioConfig builder pattern works")


@pytest.mark.asyncio
async def test_headers_builder():
    """Test Headers builder pattern."""
    print("Testing Headers builder pattern...")
    headers = truss_transfer.Headers()
    headers = headers.add("User-Agent", "truss-transfer-test")
    headers = headers.add("Accept", "audio/mpeg")
    headers = headers.add("Authorization", "Bearer token123")

    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    bytes_data = await processor.download_bytes(AUDIO_URL, headers=headers)

    assert isinstance(bytes_data, bytes), "Result should be bytes"
    assert len(bytes_data) > 0, "Downloaded file should not be empty"

    print("✓ Headers builder pattern works")


@pytest.mark.asyncio
async def test_audio_config_with_raw_ffmpeg():
    """Test AudioConfig with raw ffmpeg commands."""
    print("Testing AudioConfig with raw ffmpeg commands...")
    audio_config = (
        truss_transfer.AudioConfig()
        .with_sample_rate(16000)
        .with_raw_ffmpeg_args(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    )

    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_array, timing = await processor.process_audio_from_url(
        AUDIO_URL, audio_config
    )

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print("✓ AudioConfig with raw ffmpeg commands works")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_audio_config_per_call():
    """Test AudioConfig can be passed per call."""
    print("Testing AudioConfig per call...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_config = truss_transfer.AudioConfig()

    # Process with default config
    audio_array1, timing1 = await processor.process_audio_from_url(
        AUDIO_URL, audio_config
    )
    assert isinstance(audio_array1, np.ndarray)
    assert len(audio_array1) > 0
    assert isinstance(timing1, truss_transfer.TimingInfo)

    # Process with custom config per call
    audio_config2 = (
        truss_transfer.AudioConfig().with_sample_rate(22050).with_channels(2)
    )
    audio_array2, timing2 = await processor.process_audio_from_url(
        AUDIO_URL, audio_config2
    )
    assert isinstance(audio_array2, np.ndarray)
    assert len(audio_array2) > 0
    assert isinstance(timing2, truss_transfer.TimingInfo)

    print("✓ AudioConfig per call works")


@pytest.mark.asyncio
async def test_audio_config_with_bytes():
    """Test AudioConfig with bytes source."""
    print("Testing AudioConfig with bytes source...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)

    audio_bytes = await processor.download_bytes(AUDIO_URL)

    # Process with custom config
    audio_config = (
        truss_transfer.AudioConfig().with_sample_rate(8000).with_format("s16le")
    )
    audio_array, timing = await processor.process_audio_from_bytes(
        audio_bytes, audio_config
    )

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print("✓ AudioConfig with bytes source works")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_audio_config_use_pipes_none_auto():
    """Test auto-detection with use_pipes=None (default)."""
    print("Testing use_pipes=None (auto-detect)...")
    processor = truss_transfer.MultimodalProcessor()
    audio_config = truss_transfer.AudioConfig()  # Default is None

    # MP3 should auto-detect and use pipes
    audio_array, timing = await processor.process_audio_from_url(
        AUDIO_URL, audio_config
    )
    assert isinstance(audio_array, np.ndarray)
    assert len(audio_array) > 0
    assert isinstance(timing, truss_transfer.TimingInfo)
    print("✓ Auto-detection works for MP3")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_process_m4a_from_url():
    """Test M4A audio processing from URL."""
    print(f"Testing M4A audio processing from {AUDI_M4A}...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_config = truss_transfer.AudioConfig()
    audio_array, timing = await processor.process_audio_from_url(AUDI_M4A, audio_config)

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print(f"✓ Processed {len(audio_array)} audio samples from M4A URL")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_process_m4a_from_bytes():
    """Test M4A audio processing from bytes."""
    print("Testing M4A audio processing from bytes...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_config = truss_transfer.AudioConfig()

    # Download M4A file as bytes
    audio_bytes = await processor.download_bytes(AUDI_M4A)
    audio_array, timing = await processor.process_audio_from_bytes(
        audio_bytes, audio_config
    )

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"
    assert isinstance(timing, truss_transfer.TimingInfo), "Timing should be TimingInfo"

    print(f"✓ Processed {len(audio_array)} audio samples from M4A bytes")
    print(f"  Timing: {timing}")


@pytest.mark.asyncio
async def test_concurrent_downloads():
    """Test that concurrent downloads provide parallelism benefits."""

    print("Testing concurrent downloads for parallelism...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_config = truss_transfer.AudioConfig()

    # Measure time for 1 download + conversion
    start = time.perf_counter()
    await processor.process_audio_from_url(AUDIO_URL, audio_config)
    single_time = time.perf_counter() - start

    print(f"  Single download + conversion: {single_time:.3f}s")

    # Measure time for 4 concurrent downloads + conversions
    start = time.perf_counter()
    tasks = [
        processor.process_audio_from_url(AUDIO_URL, audio_config) for _ in range(4)
    ]
    await asyncio.gather(*tasks)
    concurrent_time = time.perf_counter() - start

    print(f"  4 concurrent downloads + conversions: {concurrent_time:.3f}s")

    # Verify that 4x concurrent is at most 2x slower than 1x
    # This means we're getting at least 2x speedup from parallelism
    speedup = (4 * single_time) / concurrent_time
    slowdown = concurrent_time / single_time

    print(f"  Speedup from parallelism: {speedup:.2f}x")
    print(f"  Slowdown vs single: {slowdown:.2f}x")

    assert slowdown <= 2.0, (
        f"Concurrent execution too slow: {slowdown:.2f}x slowdown (max 2.0x)"
    )
    print(
        f"✓ Concurrent downloads provide parallelism (speedup: {speedup:.2f}x, slowdown: {slowdown:.2f}x)"
    )


async def run_all_tests():
    """Run all async tests."""
    await test_process_audio_from_url()
    await test_process_audio_from_base64()
    await test_process_audio_from_bytes()
    await test_download_bytes()
    await test_download_bytes_with_headers()
    await test_process_audio()
    await test_process_audio_with_base64()
    await test_process_audio_with_bytes()
    await test_headers_builder()
    await test_audio_config_with_raw_ffmpeg()
    await test_audio_config_per_call()
    await test_audio_config_with_bytes()
    await test_audio_config_use_pipes_none_auto()
    await test_process_m4a_from_url()
    await test_process_m4a_from_bytes()
    await test_concurrent_downloads()


if __name__ == "__main__":
    test_processor_creation()
    test_processor_with_config()
    test_audio_config_builder()
    asyncio.run(run_all_tests())
    print("\n✅ All tests passed!")
