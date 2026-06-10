def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
N_FFT = 400
INPUT_STRIDE = 2
HOP_LENGTH = 160
CHUNK_LENGTH = 30
SAMPLE_RATE = 16000
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * INPUT_STRIDE  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
TIME_PRECISION = 1 / TOKENS_PER_SECOND
