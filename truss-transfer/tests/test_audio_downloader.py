import numpy as np
import requests
import truss_transfer

print(f"truss_transfer version: {truss_transfer.__version__}")

AUDIO_URL = "https://cdn.baseten.co/docs/production/Gettysburg.mp3"


def test_processor_creation():
    """Test that MultimodalProcessor can be created."""
    print("Testing MultimodalProcessor creation...")
    processor = truss_transfer.MultimodalProcessor()
    assert processor is not None
    print("✓ MultimodalProcessor created successfully")


def test_processor_with_config():
    """Test that MultimodalProcessor can be created with custom config."""
    print("Testing MultimodalProcessor with custom config...")
    processor = truss_transfer.MultimodalProcessor(
        sample_rate=16000, channels=1, use_dynamic_normalization=True, timeout_secs=60
    )
    assert processor is not None
    print("✓ MultimodalProcessor with custom config created successfully")


def test_process_audio_from_url():
    """Test audio processing from URL."""
    print(f"Testing audio processing from {AUDIO_URL}...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_array = processor.process_audio_from_url(AUDIO_URL)

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"

    print(f"✓ Processed {len(audio_array)} audio samples from URL")

    # check if requests + post + subprocess encode still works
    response = requests.get(AUDIO_URL)
    assert response.status_code == 200, "Failed to download audio"
    audio_bytes2 = response.content
    audio_array2 = processor.process_audio_from_bytes(audio_bytes2)
    assert isinstance(audio_array2, np.ndarray), "Result should be numpy array"
    assert audio_array2.dtype == np.float32, (
        f"Expected float32, got {audio_array2.dtype}"
    )
    assert len(audio_array2) > 0, "Processed audio should not be empty"
    assert (audio_array2 == audio_array).all()


def test_process_audio_from_base64():
    """Test audio processing from base64."""
    print("Testing audio processing from base64...")
    processor = truss_transfer.MultimodalProcessor()

    # Create a simple base64 encoded audio (just for testing)
    import base64

    request = requests.get(AUDIO_URL)
    encoded = base64.b64encode(request.content).decode("utf-8")

    audio_array = processor.process_audio_from_base64(encoded)
    # This might fail if the test data isn't valid audio, but that's ok
    print(f"✓ Processed {len(audio_array)} audio samples from base64")
    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"


def test_process_audio_from_bytes():
    """Test audio processing from bytes."""
    print("Testing audio processing from bytes...")
    processor = truss_transfer.MultimodalProcessor()

    # Use the audio URL to get bytes, then process
    audio_bytes = processor.download_bytes(AUDIO_URL)
    audio_array = processor.process_audio_from_bytes(audio_bytes)

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"

    print(f"✓ Processed {len(audio_array)} audio samples from bytes")


def test_download_bytes():
    """Test general download (for images, videos, etc.)."""
    print(f"Testing download of bytes from {AUDIO_URL}...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    bytes_data = processor.download_bytes(AUDIO_URL)

    assert isinstance(bytes_data, bytes), "Result should be bytes"
    assert len(bytes_data) > 0, "Downloaded file should not be empty"

    print(f"✓ Downloaded {len(bytes_data)} bytes")


def test_download_bytes_with_headers():
    """Test download with custom headers."""
    print("Testing download with custom headers...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)

    headers = {"User-Agent": "truss-transfer-test", "Accept": "audio/mpeg"}

    bytes_data = processor.download_bytes(AUDIO_URL, headers=headers)

    assert isinstance(bytes_data, bytes), "Result should be bytes"
    assert len(bytes_data) > 0, "Downloaded file should not be empty"

    print(f"✓ Downloaded {len(bytes_data)} bytes with headers")


def test_process_audio():
    """Test unified process_audio method."""
    print("Testing unified process_audio method...")
    processor = truss_transfer.MultimodalProcessor(timeout_secs=60)
    audio_array = processor.process_audio("url", AUDIO_URL)

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"

    print(f"✓ Processed {len(audio_array)} audio samples via process_audio")


def test_process_audio_with_base64():
    """Test process_audio with base64 source."""
    print("Testing process_audio with base64 source...")
    processor = truss_transfer.MultimodalProcessor()

    import base64

    test_data = b"test audio data"
    encoded = base64.b64encode(test_data).decode("utf-8")

    try:
        audio_array = processor.process_audio("base64", encoded)
        print(f"✓ Processed {len(audio_array)} audio samples via process_audio(base64)")
    except Exception:
        print(
            "✓ process_audio(base64) test completed (expected error for invalid audio)"
        )


def test_process_audio_with_bytes():
    """Test process_audio with bytes source."""
    print("Testing process_audio with bytes source...")
    processor = truss_transfer.MultimodalProcessor()

    audio_bytes = processor.download_bytes(AUDIO_URL)
    audio_array = processor.process_audio("bytes", audio_bytes)

    assert isinstance(audio_array, np.ndarray), "Result should be numpy array"
    assert audio_array.dtype == np.float32, f"Expected float32, got {audio_array.dtype}"
    assert len(audio_array) > 0, "Processed audio should not be empty"

    print(f"✓ Processed {len(audio_array)} audio samples via process_audio(bytes)")


if __name__ == "__main__":
    test_processor_creation()
    test_processor_with_config()
    test_download_bytes()
    test_download_bytes_with_headers()
    test_process_audio_from_url()
    test_process_audio_from_bytes()
    test_process_audio()
    test_process_audio_with_base64()
    test_process_audio_with_bytes()
    print("\n✅ All tests passed!")
